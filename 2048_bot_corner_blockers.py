#!/usr/bin/env python3
# 2048 Bot - Advanced Corner Strategy with Expert Techniques
# Improvements: Better evaluation, dynamic strategy, trap detection, endgame optimization
#
import time, json, logging, argparse, math, subprocess, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import numpy as np
import cv2
import pyautogui

# ===== Configuration =====
ACTION_DELAY = 0.55
LOOP_DELAY = 0.30
EXPECTIMAX_DEPTH = 3
GOAL_TILE = 0

PALETTE_FILE = Path("runs/color_palette.json")
UNKNOWN_RETRY = 2
UNKNOWN_RETRY_DELAY = 0.08
CENTER_CROP = (0.30, 0.70)

FUZZY_TOL = 12.0
FUZZY_MARGIN = 6.0
SESSION_CACHE_MAX = 200

DEFAULTS = {
    "4d789d": "0",
    "386489": "BORDER",
    "b1e4ef": "4",
    "82c7eb": "8",
    "6d9ddf": "16",
    "787cf8": "32",
    "db6c83": "64",
    "7abcb2": "128",
    "61a999": "256",
    "509f82": "512",
}

def hex_norm(h: str) -> str: return h.strip().lstrip('#').lower()
def bgr_to_hex(bgr: Tuple[int,int,int]) -> str:
    b,g,r = (int(bgr[0]), int(bgr[1]), int(bgr[2])); return '#%02x%02x%02x' % (r,g,b)
def hex_to_bgr(hx: str) -> Tuple[int,int,int]:
    s = hex_norm(hx); r=int(s[0:2],16); g=int(s[2:4],16); b=int(s[4:6],16); return (b,g,r)
def hsv_from_bgr(bgr: Tuple[int,int,int]) -> Tuple[float,float,float]:
    arr = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0,0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])
def hsv_circ(a: float, b: float) -> float:
    d = abs(a-b); return min(d, 180.0-d)
def hsv_dist(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    return 2.2*hsv_circ(a[0],b[0]) + abs(a[1]-b[1])/255.0*100.0 + 0.6*abs(a[2]-b[2])/255.0*100.0

def median_bg_hex(cell_bgr) -> str:
    h, w = cell_bgr.shape[:2]
    y0, y1 = int(CENTER_CROP[0]*h), int(CENTER_CROP[1]*h)
    x0, x1 = int(CENTER_CROP[0]*w), int(CENTER_CROP[1]*w)
    patch = cell_bgr[y0:y1, x0:x1] if (y1>y0 and x1>x0) else cell_bgr
    small = cv2.resize(patch, (24,24), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3,3), 0)
    med = np.median(small.reshape(-1,3), axis=0)
    b,g,r = [int(x) for x in med]
    return bgr_to_hex((b,g,r))

@dataclass
class Region:
    x:int; y:int; w:int; h:int

def ask_region() -> Region:
    print("Chọn vùng game 2048.")
    print("1) Di chuột tới GÓC TRÊN-TRÁI rồi Enter.")
    input("   Enter..."); x1,y1 = pyautogui.position(); print(f"   Top-left: ({x1},{y1})")
    print("2) Di chuột tới GÓC DƯỚI-PHẢI rồi Enter.")
    input("   Enter..."); x2,y2 = pyautogui.position(); print(f"   Bottom-right: ({x2},{y2})")
    x, y = min(x1,x2), min(y1,y2); w, h = abs(x2-x1), abs(y2-y1); side = min(w,h)
    print(f"   Vùng: x={x} y={y} w={side} h={side}")
    return Region(x,y,side,side)

def screenshot(region: Region):
    img = pyautogui.screenshot(region=(region.x, region.y, region.w, region.h))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def load_palette() -> Dict[str,str]:
    if PALETTE_FILE.exists():
        try:
            data = json.loads(PALETTE_FILE.read_text(encoding="utf-8"))
            return {**DEFAULTS, **{hex_norm(k): str(v) for k,v in data.get("colors",{}).items()}}
        except Exception: pass
    return DEFAULTS.copy()

def save_palette(pal: Dict[str,str]):
    PALETTE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PALETTE_FILE.write_text(json.dumps({"colors": pal}, indent=2), encoding="utf-8")

class HexFuzzyRecognizer:
    def __init__(self, palette: Dict[str,str], fuzzy_tol=FUZZY_TOL, fuzzy_margin=FUZZY_MARGIN, enable_fuzzy=True):
        self.colors = {('#'+hex_norm(k)): str(v) for k,v in palette.items()}
        self.entries = [(hx, hsv_from_bgr(hex_to_bgr(hx)), v) for hx,v in self.colors.items()]
        self.border_hex = next((hx for hx,v in self.colors.items() if v.upper()=="BORDER"), None)
        self.fuzzy_tol = float(fuzzy_tol); self.fuzzy_margin=float(fuzzy_margin)
        self.enable_fuzzy = bool(enable_fuzzy); self.cache = {}

    def _resolve(self, hex_bg: str) -> Optional[str]:
        if hex_bg in self.colors: return self.colors[hex_bg]
        if hex_bg in self.cache: return self.cache[hex_bg]
        if not self.enable_fuzzy: return None
        hsv = hsv_from_bgr(hex_to_bgr(hex_bg))
        best=(None,1e9,None); second=(None,1e9,None)
        for hx, hsv_ref, val in self.entries:
            d=hsv_dist(hsv, hsv_ref)
            if d<best[1]: second=best; best=(hx,d,val)
            elif d<second[1]: second=(hx,d,val)
        if best[0] and best[1]<=self.fuzzy_tol and (second[1]-best[1])>=self.fuzzy_margin:
            self.cache[hex_bg]=best[2]
            if len(self.cache)>SESSION_CACHE_MAX: self.cache.pop(next(iter(self.cache)))
            return best[2]
        return None

    def detect_board_once(self, img_bgr, region: Region):
        h, w = img_bgr.shape[:2]; ch, cw = h//4, w//4
        board = [[0]*4 for _ in range(4)]; unknowns=[]
        for i in range(4):
            for j in range(4):
                y1, x1 = i*ch, j*cw
                cell = img_bgr[y1:y1+ch, x1:x1+cw]
                if cell.size == 0: continue
                hex_bg = median_bg_hex(cell)
                if self.border_hex and hex_bg == self.border_hex:
                    board[i][j] = 0; continue
                v = self._resolve(hex_bg)
                if v is None:
                    rect=(region.x+x1, region.y+y1, region.x+x1+cw, region.y+y1+ch)
                    unknowns.append({"i":i,"j":j,"hex":hex_bg,"rect":rect})
                else:
                    board[i][j] = 0 if v.upper()=="BORDER" else int(v) if v.isdigit() else 0
        return board, unknowns

    def detect_board(self, region: Region):
        img = screenshot(region)
        board, unknowns = self.detect_board_once(img, region)
        if not unknowns: return board, []
        for _ in range(UNKNOWN_RETRY):
            time.sleep(UNKNOWN_RETRY_DELAY)
            img2 = screenshot(region)
            board2, unknowns2 = self.detect_board_once(img2, region)
            if not unknowns2: return board2, []
            board, unknowns = board2, unknowns2
        return board, unknowns

class AdvancedBot:
    def __init__(self, region: Region, recog: HexFuzzyRecognizer, out_dir: Path, autoplay=True, input_mode="keys", driver="pydirectinput", adb_serial=None, swipe_ratio=0.6):
        self.region=region; self.recog=recog; self.autoplay=autoplay
        self.input_mode=input_mode; self.driver=driver; self.swipe_ratio=max(0.2,min(1.0,swipe_ratio))
        self.adb_serial=adb_serial
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log", encoding="utf-8")])
        self.log=logging.getLogger("bot"); self.step=0
        self.prev_board=None; self.prev_move=None
        
        # Strategy configuration
        self._corner = "tr"
        self._anti_down_strict = True
        self._move_history = []  # Track recent moves for pattern detection
        self._trap_detection_enabled = True

    def pretty(self, b): return "\n".join("|".join(f"{(x if x else 0):4d}" for x in r) for r in b)

    # 2048 mechanics
    def _move_left(self, b):
        nb=[r[:] for r in b]; moved=False
        for i in range(4):
            nums=[x for x in nb[i] if x]; merged=[]; k=0
            while k<len(nums):
                if k+1<len(nums) and nums[k]==nums[k+1]: merged.append(nums[k]*2); k+=2
                else: merged.append(nums[k]); k+=1
            merged+=[0]*(4-len(merged)); moved|=(merged!=nb[i]); nb[i]=merged
        return nb,moved
    def _move_right(self,b): rb=[r[::-1] for r in b]; nb,m=self._move_left(rb); return [r[::-1] for r in nb],m
    def _move_up(self,b): t=[list(x) for x in zip(*b)]; nb,m=self._move_left(t); return [list(x) for x in zip(*nb)],m
    def _move_down(self,b): t=[list(x) for x in zip(*b)]; nb,m=self._move_right(t); return [list(x) for x in zip(*nb)],m
    def _apply(self,m,b): return {"←":self._move_left,"→":self._move_right,"↑":self._move_up,"↓":self._move_down}.get(m,lambda x:(x,False))(b)

    def _snake_path(self, corner):
        return {
            'tl': [(0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2),(2,3),(3,3),(3,2),(3,1),(3,0)],
            'tr': [(0,3),(0,2),(0,1),(0,0),(1,0),(1,1),(1,2),(1,3),(2,3),(2,2),(2,1),(2,0),(3,0),(3,1),(3,2),(3,3)],
            'bl': [(3,0),(3,1),(3,2),(3,3),(2,3),(2,2),(2,1),(2,0),(1,0),(1,1),(1,2),(1,3),(0,3),(0,2),(0,1),(0,0)],
            'br': [(3,3),(3,2),(3,1),(3,0),(2,0),(2,1),(2,2),(2,3),(1,3),(1,2),(1,1),(1,0),(0,0),(0,1),(0,2),(0,3)],
        }[corner]

    # Advanced evaluation functions
    def _weighted_sum_score(self, b):
        """Tính điểm theo trọng số vị trí với corner strategy"""
        weights = {
            'tr': [[65536, 32768, 16384, 8192],
                   [512, 1024, 2048, 4096], 
                   [256, 128, 64, 32],
                   [4, 8, 16, 2]],
            'tl': [[8192, 16384, 32768, 65536],
                   [4096, 2048, 1024, 512],
                   [32, 64, 128, 256],
                   [2, 16, 8, 4]]
        }
        w = weights.get(self._corner, weights['tr'])
        return sum(b[i][j] * w[i][j] for i in range(4) for j in range(4))

    def _gradient_score(self, b):
        """Đánh giá gradient - ưu tiên giá trị giảm dần từ corner"""
        score = 0
        # Horizontal gradients
        for i in range(4):
            for j in range(3):
                if self._corner in ['tr', 'br']:
                    if b[i][j+1] != 0 and b[i][j] != 0:
                        score += (b[i][j+1] - b[i][j]) * (4-j)
                else:
                    if b[i][j] != 0 and b[i][j+1] != 0:
                        score += (b[i][j] - b[i][j+1]) * (4-j)
        
        # Vertical gradients
        for j in range(4):
            for i in range(3):
                if self._corner in ['tl', 'tr']:
                    if b[i][j] != 0 and b[i+1][j] != 0:
                        score += (b[i][j] - b[i+1][j]) * (4-i)
                else:
                    if b[i+1][j] != 0 and b[i][j] != 0:
                        score += (b[i+1][j] - b[i][j]) * (4-i)
        return score

    def _monotonicity_score(self, b):
        """Đánh giá tính đơn điệu của board"""
        mono_score = 0
        
        # Check rows
        for row in b:
            # Left to right monotonicity
            increasing = decreasing = 0
            for i in range(3):
                if row[i] != 0 and row[i+1] != 0:
                    if row[i] < row[i+1]:
                        increasing += row[i+1] - row[i]
                    elif row[i] > row[i+1]:
                        decreasing += row[i] - row[i+1]
            mono_score += max(increasing, decreasing)
        
        # Check columns
        for j in range(4):
            col = [b[i][j] for i in range(4)]
            increasing = decreasing = 0
            for i in range(3):
                if col[i] != 0 and col[i+1] != 0:
                    if col[i] < col[i+1]:
                        increasing += col[i+1] - col[i]
                    elif col[i] > col[i+1]:
                        decreasing += col[i] - col[i+1]
            mono_score += max(increasing, decreasing)
        
        return mono_score

    def _merge_potential_advanced(self, b):
        """Cải tiến merge potential với trọng số theo vị trí"""
        potential = 0
        weights = [[16, 8, 4, 2], [8, 4, 2, 1], [4, 2, 1, 0.5], [2, 1, 0.5, 0.25]]
        
        # Horizontal merges
        for i in range(4):
            for j in range(3):
                if b[i][j] != 0 and b[i][j] == b[i][j+1]:
                    potential += b[i][j] * weights[i][j]
        
        # Vertical merges
        for i in range(3):
            for j in range(4):
                if b[i][j] != 0 and b[i][j] == b[i+1][j]:
                    potential += b[i][j] * weights[i][j]
        
        return potential

    def _corner_protection_score(self, b):
        """Bảo vệ corner - phạt nếu max tile không ở corner"""
        max_val = max(max(row) for row in b)
        corners = {'tl': (0,0), 'tr': (0,3), 'bl': (3,0), 'br': (3,3)}
        corner_pos = corners[self._corner]
        
        if b[corner_pos[0]][corner_pos[1]] == max_val:
            return 1000  # Bonus for max in corner
        else:
            # Penalty proportional to distance from corner
            for i in range(4):
                for j in range(4):
                    if b[i][j] == max_val:
                        distance = abs(i - corner_pos[0]) + abs(j - corner_pos[1])
                        return -distance * 200
        return 0

    def _trap_detection(self, b):
        """Phát hiện trap patterns - các vị trí có thể gây kẹt"""
        penalty = 0
        
        # Check for small tiles in corner edges
        if self._corner == 'tr':
            # Check top row and right column
            top_row = b[0]
            right_col = [b[i][3] for i in range(4)]
            
            # Penalize small values breaking monotonicity
            for i in range(1, 4):
                if top_row[i] != 0 and top_row[i-1] != 0:
                    if top_row[i] > top_row[i-1] * 4:  # Big jump
                        penalty += 50
            
            for i in range(1, 4):
                if right_col[i] != 0 and right_col[i-1] != 0:
                    if right_col[i] > right_col[i-1] * 4:  # Big jump
                        penalty += 50
        
        return -penalty

    def _endgame_optimization(self, b):
        """Tối ưu hóa cho endgame khi board gần full"""
        empty_count = sum(1 for row in b for cell in row if cell == 0)
        
        if empty_count <= 3:  # Endgame
            # Prioritize moves that create merges
            merge_bonus = 0
            for i in range(4):
                for j in range(3):
                    if b[i][j] != 0 and b[i][j] == b[i][j+1]:
                        merge_bonus += b[i][j] * 2
            for i in range(3):
                for j in range(4):
                    if b[i][j] != 0 and b[i][j] == b[i+1][j]:
                        merge_bonus += b[i][j] * 2
            return merge_bonus * 3  # Triple bonus in endgame
        
        return 0

    def _dynamic_depth_adjustment(self, b):
        """Điều chỉnh độ sâu search theo tình huống"""
        empty_count = sum(1 for row in b for cell in row if cell == 0)
        max_tile = max(max(row) for row in b)
        
        if empty_count <= 4:  # Critical situation
            return min(EXPECTIMAX_DEPTH + 2, 6)
        elif max_tile >= 1024:  # Late game
            return EXPECTIMAX_DEPTH + 1
        else:
            return EXPECTIMAX_DEPTH

    def _eval(self, b):
        """Hàm đánh giá cải tiến với nhiều yếu tố"""
        empty = sum(1 for r in b for c in r if c == 0)
        max_val = max(max(r) for r in b) if any(any(r) for r in b) else 0
        
        # Core scores
        empty_score = empty * 150
        max_score = math.log2(max_val + 1) * 80
        weighted_sum = self._weighted_sum_score(b) * 0.001
        monotonicity = self._monotonicity_score(b) * 0.5
        merge_potential = self._merge_potential_advanced(b)
        corner_protection = self._corner_protection_score(b)
        
        # Advanced features
        gradient_score = self._gradient_score(b) * 0.3
        trap_penalty = self._trap_detection(b) if self._trap_detection_enabled else 0
        endgame_bonus = self._endgame_optimization(b)
        
        total_score = (empty_score + max_score + weighted_sum + monotonicity + 
                      merge_potential + corner_protection + gradient_score + 
                      trap_penalty + endgame_bonus)
        
        return total_score

    def _order_moves_dynamic(self, b):
        """Dynamic move ordering based on board state"""
        base_order = {
            'tr': ["→", "↑", "←", "↓"],
            'tl': ["←", "↑", "→", "↓"],
            'br': ["→", "↓", "←", "↑"],
            'bl': ["←", "↓", "→", "↑"]
        }
        
        order = base_order.get(self._corner, base_order['tr'])
        
        # Adjust based on recent moves to avoid cycles
        if len(self._move_history) >= 3:
            recent = self._move_history[-3:]
            if len(set(recent)) <= 2:  # Possible cycle
                # Shuffle the order slightly
                if order[0] in recent:
                    order = order[1:] + [order[0]]
        
        return order

    def _expectimax(self, b, depth, player):
        """Expectimax với dynamic depth adjustment"""
        if depth == 0:
            return self._eval(b)
            
        if player:  # AI turn
            best = -1e18
            moves = self._order_moves_dynamic(b)
            
            # Anti-down strategy with exceptions
            if self._anti_down_strict and depth == self._dynamic_depth_adjustment(b):
                empty_count = sum(1 for row in b for cell in row if cell == 0)
                if empty_count > 2:  # Only avoid down if we have options
                    moves = [m for m in moves if m != "↓"]
            
            for move in moves:
                new_board, moved = self._apply(move, b)
                if moved:
                    score = self._expectimax(new_board, depth - 1, False)
                    best = max(best, score)
            
            return best if best > -1e17 else self._eval(b)
        else:  # Random tile placement
            empty_cells = [(i, j) for i in range(4) for j in range(4) if b[i][j] == 0]
            if not empty_cells:
                return self._eval(b)
            
            expected = 0.0
            for i, j in empty_cells:
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    new_board = [row[:] for row in b]
                    new_board[i][j] = value
                    tile_score = self._expectimax(new_board, depth - 1, True)
                    expected += prob * (1.0 / len(empty_cells)) * tile_score
            
            return expected

    def best_move(self, board):
        """Chọn nước đi tốt nhất với chiến thuật cải tiến"""
        depth = self._dynamic_depth_adjustment(board)
        moves = self._order_moves_dynamic(board)
        
        # Get valid moves
        valid_moves = []
        for move in moves:
            new_board, moved = self._apply(move, board)
            if moved:
                valid_moves.append((move, new_board))
        
        if not valid_moves:
            return "?"
        
        # Apply anti-down strategy
        if self._anti_down_strict and len(valid_moves) > 1:
            non_down_moves = [(m, b) for (m, b) in valid_moves if m != "↓"]
            if non_down_moves:
                valid_moves = non_down_moves
        
        # Evaluate moves
        best_move = "?"
        best_score = -1e18
        
        for move, new_board in valid_moves:
            score = self._expectimax(new_board, depth - 1, False)
            
            # Add small bonus for preferred moves to break ties
            if move == moves[0]:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Update move history
        self._move_history.append(best_move)
        if len(self._move_history) > 10:
            self._move_history.pop(0)
        
        return best_move

    # Input methods (unchanged)
    def _press_pyautogui(self, move):
        key_map={"←":"left","→":"right","↑":"up","↓":"down"}
        cx=self.region.x+self.region.w//2; cy=self.region.y+self.region.h//2
        pyautogui.click(cx,cy); time.sleep(0.03); pyautogui.press(key_map[move])
    def _press_pydirectinput(self, move):
        import pydirectinput
        key_map={"←":"left","→":"right","↑":"up","↓":"down"}
        pydirectinput.PAUSE=0; pydirectinput.press(key_map[move])
    def _press_keyboard(self, move):
        import keyboard
        key_map={"←":"left","→":"right","↑":"up","↓":"down"}
        keyboard.send(key_map[move])
    def _press_adb(self, move):
        map_adb={"←":"DPAD_LEFT","→":"DPAD_RIGHT","↑":"DPAD_UP","↓":"DPAD_DOWN"}
        key = map_adb[move]
        adb = shutil.which("adb")
        if adb is None:
            print("adb không có trong PATH"); return
        cmd = [adb]
        if self.adb_serial: cmd += ["-s", self.adb_serial]
        cmd += ["shell", "input", "keyevent", key]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def press(self, move):
        if self.input_mode=="swipe":
            cx=self.region.x+self.region.w//2; cy=self.region.y+self.region.h//2
            dx=self.region.w//2; dy=self.region.h//2; dist=int(0.6*min(dx,dy))
            if move=="←": start=(cx+dist//2, cy); end=(cx-dist, cy)
            elif move=="→": start=(cx-dist//2, cy); end=(cx+dist, cy)
            elif move=="↑": start=(cx, cy+dist//2); end=(cx, cy-dist)
            else: start=(cx, cy-dist//2); end=(cx, cy+dist)
            pyautogui.moveTo(*start); pyautogui.dragTo(*end, duration=0.18); return
        if self.driver=="pydirectinput": self._press_pydirectinput(move)
        elif self.driver=="keyboard": self._press_keyboard(move)
        elif self.driver=="adb": self._press_adb(move)
        else: self._press_pyautogui(move)

    # Auto-learning methods (unchanged)
    def _auto_learn_from_merge(self, pre_board, move, post_unknowns):
        exp_board,_ = self._apply(move, pre_board)
        if not post_unknowns: return False
        pal = load_palette(); updated=False
        for u in post_unknowns:
            i,j,hx = u["i"],u["j"],u["hex"]
            val = exp_board[i][j]
            if val>0:
                key = hex_norm(hx)
                if key not in pal:
                    pal[key]=str(val); updated=True
                    print(f"[auto-learn] {hx} -> {val} tại (row={i}, col={j})")
        if updated:
            save_palette(pal)
            self.recog = HexFuzzyRecognizer(pal, enable_fuzzy=True, fuzzy_tol=FUZZY_TOL, fuzzy_margin=FUZZY_MARGIN)
        return updated

    def _is_stuck(self, board):
        for m in ("←","→","↑","↓"):
            _, moved = self._apply(m, board)
            if moved: return False
        return True

    def loop(self):
        self.log.info("Starting Advanced 2048 Bot...")
        while True:
            board, unknowns = self.recog.detect_board(self.region)
            self.step += 1
            
            # Auto-learning from previous move
            if self.prev_board is not None and self.prev_move is not None and unknowns:
                if self._auto_learn_from_merge(self.prev_board, self.prev_move, unknowns):
                    board, unknowns = self.recog.detect_board(self.region)
            
            # Handle unknown colors
            if unknowns:
                print("\nMÀU CHƯA BIẾT:")
                for u in unknowns:
                    x1,y1,x2,y2 = u["rect"]
                    print(f"- (row={u['i']}, col={u['j']}) HEX={u['hex']} Rect={x1,y1,x2,y2}")
                hex_in = input("Map thêm (a=b,c=d) hoặc Enter để dừng: ").strip()
                if hex_in:
                    pal=load_palette()
                    for pair in hex_in.split(","):
                        if "=" in pair:
                            k,v = pair.split("=",1); pal[hex_norm(k)] = v.strip()
                    save_palette(pal)
                    self.recog = HexFuzzyRecognizer(pal, enable_fuzzy=True, fuzzy_tol=FUZZY_TOL, fuzzy_margin=FUZZY_MARGIN)
                else:
                    print("Dừng."); return
                continue

            # Get best move using advanced strategy
            move = self.best_move(board)
            
            # Calculate statistics
            max_val = max(max(r) for r in board) if any(any(r) for r in board) else 0
            empty = sum(1 for r in board for c in r if c == 0)
            total = sum(sum(r) for r in board)
            
            # Enhanced logging with strategy info
            self.log.info("STEP %d | corner=%s | move=%s | max=%d | empty=%d | sum=%d | depth=%d\n%s",
                          self.step, self._corner, move, max_val, empty, total, 
                          self._dynamic_depth_adjustment(board), self.pretty(board))

            # Check win condition
            if GOAL_TILE > 0 and max_val >= GOAL_TILE:
                self.log.info("VICTORY! Reached %d (goal=%d)", max_val, GOAL_TILE)
                return
                
            # Check game over
            if move == "?" and self._is_stuck(board):
                self.log.info("GAME OVER | Final score: max=%d, sum=%d", max_val, total)
                return

            # Execute move
            self.prev_board = [r[:] for r in board]
            self.prev_move = move
            
            if self.autoplay and move != "?":
                self.press(move)
                time.sleep(ACTION_DELAY)
            
            time.sleep(LOOP_DELAY)

def main():
    ap = argparse.ArgumentParser(description="Advanced 2048 Bot with Expert Strategy")
    ap.add_argument('--autoplay', action='store_true', default=True, help='Auto play game')
    ap.add_argument('--input', choices=['keys','swipe'], default='keys', help='Input method')
    ap.add_argument('--driver', choices=['pyautogui','pydirectinput','keyboard','adb'], default='keyboard', help='Input driver')
    ap.add_argument('--adb-serial', type=str, default=None, help='ADB device serial')
    ap.add_argument('--swipe-ratio', type=float, default=0.6, help='Swipe distance ratio')
    ap.add_argument('--depth', type=int, default=4, help='Base expectimax search depth')
    ap.add_argument('--goal', type=int, default=0, help='Target tile value (0 = no limit)')
    ap.add_argument('--corner', choices=['tl','tr','bl','br'], default='tr', help='Corner strategy')
    ap.add_argument('--disable-trap-detection', action='store_true', help='Disable trap pattern detection')
    
    args = ap.parse_args()
    
    # Set global variables
    global EXPECTIMAX_DEPTH, GOAL_TILE
    EXPECTIMAX_DEPTH = args.depth
    GOAL_TILE = args.goal
    
    # Setup output directory
    out_dir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    
    # Initialize components
    region = ask_region()
    palette = load_palette()
    recognizer = HexFuzzyRecognizer(palette, enable_fuzzy=True, fuzzy_tol=FUZZY_TOL, fuzzy_margin=FUZZY_MARGIN)
    
    # Create advanced bot
    bot = AdvancedBot(region, recognizer, out_dir, autoplay=args.autoplay, 
                     input_mode=args.input, driver=args.driver, 
                     adb_serial=args.adb_serial, swipe_ratio=args.swipe_ratio)
    
    # Configure bot strategy
    bot._corner = args.corner
    bot._trap_detection_enabled = not args.disable_trap_detection
    
    print(f"\n=== Advanced 2048 Bot Configuration ===")
    print(f"Corner Strategy: {args.corner.upper()}")
    print(f"Base Search Depth: {args.depth}")
    print(f"Goal Tile: {args.goal if args.goal > 0 else 'No limit'} (65536 = 2^16)")
    print(f"Trap Detection: {'Enabled' if bot._trap_detection_enabled else 'Disabled'}")
    print(f"Anti-Down Strategy: {'Enabled' if bot._anti_down_strict else 'Disabled'}")
    print(f"Input Method: {args.input} ({args.driver})")
    print("=" * 40)
    print("Target: Reach 65536 tile (2^16) - Ultimate Challenge! 🎯")
    print("Power-ups: 🔨Hammer(2) 💣Bomb(1) 🪄Wand(3)")
    print("=" * 40)
    
    # Start the bot
    try:
        bot.loop()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot encountered an error: {e}")
        raise

if __name__ == "__main__":
    main()