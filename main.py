import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import datetime
import os
import math
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HAND_MODEL_PATH = "hand_landmarker.task"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PANEL_HEIGHT = 84
TOOLBAR_WIDTH = 92
BRUSH_SIZE = 10
ERASER_SIZE = 34
SMOOTHING = 6
OUTPUT_DIR = "drawings"
PINCH_DIST = 42
DEFAULT_SHAPE_SIZE = 26
GROUP_HOLD_FRAMES = 20

PALETTE = [
    (0, 0, 255),
    (0, 130, 255),
    (0, 255, 255),
    (80, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 200),
    (255, 0, 128),
    None,
]

SHAPE_KINDS = ["circle", "square", "rectangle", "triangle", "pentagon", "hexagon", "star", "heart", "line"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand_skeleton(frame, landmarks):
    pts = [(int(lm.x * FRAME_WIDTH), int(lm.y * FRAME_HEIGHT)) for lm in landmarks]
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 4, (255, 255, 0), -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)


def rect_contains(rect, x, y):
    x1, y1, x2, y2 = rect[:4]
    return x1 <= x <= x2 and y1 <= y <= y2


def rounded_rect(img, p1, p2, radius, color, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def center_text(img, text, x1, y1, x2, y2, scale=0.5, color=(255, 255, 255), thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def shade(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def draw_button(img, rect, label, color, active=False, text_color=(255, 255, 255), scale=0.5):
    x1, y1, x2, y2 = rect
    r = min(8, (y2 - y1) // 2)
    rounded_rect(img, (x1 + 2, y1 + 3), (x2 + 2, y2 + 3), r, (10, 10, 14), -1)
    rounded_rect(img, (x1, y1), (x2, y2), r, color, -1)
    rounded_rect(img, (x1 + 4, y1 + 2), (x2 - 4, y1 + max(5, (y2 - y1) // 3)), 3, shade(color, 1.45), -1)
    rounded_rect(img, (x1 + 3, y2 - 4), (x2 - 3, y2 - 1), 2, shade(color, 0.55), -1)
    if active:
        rounded_rect(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), r + 2, shade(color, 1.3), 2)
        rounded_rect(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), r + 3, (255, 255, 255), 1)
    else:
        rounded_rect(img, (x1, y1), (x2, y2), r, (255, 255, 255), 1)
    center_text(img, label, x1, y1, x2, y2, scale, text_color, 2)


def smooth(points):
    pts = list(points)
    if not pts:
        return None
    x = int(np.mean([p[0] for p in pts]))
    y = int(np.mean([p[1] for p in pts]))
    return x, y


def fingers_up(hand):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [int(hand[t].y < hand[p].y) for t, p in zip(tips, pips)]


def rainbow_bgr(step):
    h = int(step) % 180
    px = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def regular_polygon(center, radius, n, rotation=-math.pi / 2):
    cx, cy = center
    pts = []
    for i in range(n):
        ang = rotation + 2 * math.pi * i / n
        pts.append((int(cx + radius * math.cos(ang)), int(cy + radius * math.sin(ang))))
    return np.array(pts, np.int32).reshape((-1, 1, 2))


def star_points(center, outer):
    cx, cy = center
    inner = outer * 0.45
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = outer if i % 2 == 0 else inner
        pts.append((int(cx + rad * math.cos(ang)), int(cy + rad * math.sin(ang))))
    return np.array(pts, np.int32).reshape((-1, 1, 2))


def heart_points(center, size):
    cx, cy = center
    pts = []
    for t in range(0, 360, 4):
        a = math.radians(t)
        x = 16 * math.sin(a) ** 3
        y = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
        pts.append((int(cx + size * x / 16), int(cy - size * (y + 2) / 16)))
    return np.array(pts, np.int32).reshape((-1, 1, 2))


def draw_shape(canvas, alpha, kind, cx, cy, size, color, outline=None, thickness=2, fill=True):
    if kind == "circle":
        if fill:
            if alpha is not None:
                cv2.circle(alpha, (cx, cy), size, 255, -1)
            cv2.circle(canvas, (cx, cy), size, color, -1)
        if outline:
            cv2.circle(canvas, (cx, cy), size, outline, thickness)
    elif kind == "square":
        if fill:
            if alpha is not None:
                cv2.rectangle(alpha, (cx - size, cy - size), (cx + size, cy + size), 255, -1)
            cv2.rectangle(canvas, (cx - size, cy - size), (cx + size, cy + size), color, -1)
        if outline:
            cv2.rectangle(canvas, (cx - size, cy - size), (cx + size, cy + size), outline, thickness)
    elif kind == "rectangle":
        hw = int(size * 1.6)
        hh = size
        if fill:
            if alpha is not None:
                cv2.rectangle(alpha, (cx - hw, cy - hh), (cx + hw, cy + hh), 255, -1)
            cv2.rectangle(canvas, (cx - hw, cy - hh), (cx + hw, cy + hh), color, -1)
        if outline:
            cv2.rectangle(canvas, (cx - hw, cy - hh), (cx + hw, cy + hh), outline, thickness)
    elif kind in ("triangle", "pentagon", "hexagon"):
        n = {"triangle": 3, "pentagon": 5, "hexagon": 6}[kind]
        pts = regular_polygon((cx, cy), size * 1.35, n)
        if fill:
            if alpha is not None:
                cv2.fillPoly(alpha, [pts], 255)
            cv2.fillPoly(canvas, [pts], color)
        if outline:
            cv2.polylines(canvas, [pts], True, outline, thickness)
    elif kind == "star":
        pts = star_points((cx, cy), size * 1.5)
        if fill:
            if alpha is not None:
                cv2.fillPoly(alpha, [pts], 255)
            cv2.fillPoly(canvas, [pts], color)
        if outline:
            cv2.polylines(canvas, [pts], True, outline, thickness)
    elif kind == "heart":
        pts = heart_points((cx, cy), size * 1.9)
        if fill:
            if alpha is not None:
                cv2.fillPoly(alpha, [pts], 255)
            cv2.fillPoly(canvas, [pts], color)
        if outline:
            cv2.polylines(canvas, [pts], True, outline, thickness)
    elif kind == "line":
        half = int(size * 1.5)
        lw = max(3, size // 3)
        if fill:
            if alpha is not None:
                cv2.line(alpha, (cx - half, cy), (cx + half, cy), 255, lw)
            cv2.line(canvas, (cx - half, cy), (cx + half, cy), color, lw)
        if outline:
            cv2.line(canvas, (cx - half, cy), (cx + half, cy), outline, thickness)


def shape_bbox(kind, cx, cy, size):
    if kind == "rectangle":
        hw = int(size * 1.6)
        return cx - hw, cy - size, cx + hw, cy + size
    if kind == "line":
        half = int(size * 1.5)
        return cx - half, cy - size, cx + half, cy + size
    if kind in ("triangle", "pentagon", "hexagon"):
        r = int(size * 1.35)
    elif kind == "star":
        r = int(size * 1.5)
    elif kind == "heart":
        r = int(size * 1.9)
    else:
        r = size
    return cx - r, cy - r, cx + r, cy + r


class VirtualWritingBoard:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not os.path.exists(HAND_MODEL_PATH):
            raise RuntimeError(
                f"Hand model not found: {HAND_MODEL_PATH}\n"
                "Download it from:\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_counter = 0

        self.color = PALETTE[0]
        self.hue = 0
        self.tool = "draw"
        self.mode = "none"
        self.canvas = None
        self.alpha = None
        self.shape_layer = None
        self.shape_alpha = None
        self.frame = None

        self.draw_history = deque(maxlen=SMOOTHING)
        self.erase_history = deque(maxlen=SMOOTHING)
        self.prev_draw = None
        self.prev_erase = None
        self.prev_shape = None

        self.cursor_point = None
        self.cursor_radius = 0

        self.shapes = []
        self.next_id = 1
        self.selected_ids = set()
        self.groups = {}
        self.next_group_id = 1
        self.current_shape = "circle"
        self.dragging = False
        self.drag_prev = None
        self.pinch_active_prev = False
        self.pinch_dist_prev = None
        self.group_hold = 0
        self.group_action = None

        self.saved_once = False
        self.clear_once = False
        self.exit_once = False
        self.group_once = False
        self.ungroup_once = False
        self.tool_once = False
        self.exit_requested = False

        self.swatches = []
        self.shape_slots = []
        self.button_rects = {}
        self.tool_rects = {}
        self.group_rects = {}
        self._build_panel_layout()

    def _build_panel_layout(self):
        t_w, t_h = 84, 30
        ty = 7
        self.tool_rects = {}
        for i, name in enumerate(["DRAW", "SHAPE", "SELECT"]):
            x = 104 + i * (t_w + 8)
            self.tool_rects[name] = (x, ty, x + t_w, ty + t_h)

        a_w, a_h = 82, 30
        ay = 7
        base = FRAME_WIDTH - a_w - 12
        self.button_rects = {}
        for i, name in enumerate(["SAVE", "CLEAR", "EXIT"]):
            x = base - (2 - i) * (a_w + 8)
            self.button_rects[name] = (x, ay, x + a_w, ay + a_h)

        r = 12
        spacing = 42
        x0 = 164
        cy = 60
        self.swatches = []
        for i, color in enumerate(PALETTE):
            cx = x0 + i * spacing
            self.swatches.append(((cx - r, cy - r, cx + r, cy + r), color))

        g_w, g_h = 84, 32
        gy = 44
        gx = FRAME_WIDTH - g_w - 12
        self.group_rects = {
            "UNGROUP": (gx, gy, gx + g_w, gy + g_h),
            "GROUP": (gx - g_w - 8, gy, gx - 8, gy + g_h),
        }

        cell_h = 74
        sy = 26
        self.shape_slots = []
        for i, kind in enumerate(SHAPE_KINDS):
            self.shape_slots.append((0, sy + i * cell_h, TOOLBAR_WIDTH, sy + (i + 1) * cell_h, kind))

    def run(self):
        try:
            while True:
                ok, raw = self.cap.read()
                if not ok:
                    break

                raw = cv2.flip(raw, 1)
                self.frame = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))

                if self.canvas is None:
                    h, w = self.frame.shape[:2]
                    self.canvas = np.zeros((h, w, 3), np.uint8)
                    self.alpha = np.zeros((h, w), np.uint8)
                    self.shape_layer = np.zeros((h, w, 3), np.uint8)
                    self.shape_alpha = np.zeros((h, w), np.uint8)

                self.frame_counter += 1
                detect_img = cv2.resize(self.frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))
                rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(mp_image, int(self.frame_counter * 33))

                self.draw_ui()

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    self.handle_gesture(hand)
                else:
                    self._reset_tracking()

                overlay = self.compose()

                cv2.imshow("Virtual Writing Board", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or self.exit_requested:
                    break
        finally:
            self.cap.release()
            self.landmarker.close()
            cv2.destroyAllWindows()

    def handle_gesture(self, hand):
        draw_hand_skeleton(self.frame, hand)
        fingers = fingers_up(hand)
        index_tip = self.to_pixel(hand[8])
        middle_tip = self.to_pixel(hand[12])
        thumb_tip = self.to_pixel(hand[4])
        thumb_up = hand[4].y < hand[3].y
        cv2.circle(self.frame, index_tip, 6, (0, 255, 0), -1)

        n_up = sum(fingers)
        pinch = (
            math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]) < PINCH_DIST
            and n_up <= 1
        )
        three_up = fingers[0] and fingers[1] and fingers[2] and not fingers[3]

        if n_up >= 4:
            self.mode = "clear"
            self._reset_tracking()
            self.clear_canvas()
            return

        if pinch:
            self.mode = "select"
            self._reset_stroke_state()
            self._reset_group_gesture()
            pd = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
            self.handle_selection(
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2,
                pd,
            )
            return

        self._reset_pinch()

        if three_up:
            self.mode = "ungroup" if thumb_up else "group"
            self.handle_group_gesture(self.mode)
            return

        self._reset_group_gesture()

        if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
            self.mode = "erase"
            self.prev_draw = None
            self.erase(index_tip, middle_tip)
            return

        if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]:
            if self.tool == "shape":
                self.mode = "shape"
                self.place_shape(index_tip)
            elif self.tool == "select":
                self.mode = "select"
                if self.in_ui_region(index_tip):
                    self.handle_panel(index_tip)
                else:
                    self.select_tap(index_tip)
            else:
                self.mode = "draw"
                self.draw(index_tip)
            return

        self.mode = "none"
        self._reset_tracking()

    def draw(self, index_tip):
        self.draw_history.append(index_tip)
        point = smooth(self.draw_history)
        if point is None:
            return
        self.cursor_point = point
        self.cursor_radius = BRUSH_SIZE // 2

        if self.in_ui_region(point):
            self.prev_draw = None
            self.handle_panel(point)
            return

        self.reset_click_flags()
        color = self.paint_color()

        if self.prev_draw is not None:
            cv2.line(self.canvas, self.prev_draw, point, color, BRUSH_SIZE)
            cv2.line(self.alpha, self.prev_draw, point, 255, BRUSH_SIZE)
        cv2.circle(self.canvas, point, BRUSH_SIZE // 2, color, -1)
        cv2.circle(self.alpha, point, BRUSH_SIZE // 2, 255, -1)
        self.prev_draw = point

    def place_shape(self, index_tip):
        self.draw_history.append(index_tip)
        point = smooth(self.draw_history)
        if point is None:
            return
        self.cursor_point = point
        self.cursor_radius = DEFAULT_SHAPE_SIZE

        if self.in_ui_region(point):
            self.prev_shape = None
            self.handle_panel(point)
            return

        self.reset_click_flags()
        if self.prev_shape is None:
            obj = {
                "id": self.next_id,
                "kind": self.current_shape,
                "center": (point[0], point[1]),
                "size": DEFAULT_SHAPE_SIZE,
                "color": self.paint_color(),
                "group": None,
            }
            self.next_id += 1
            self.shapes.append(obj)
            self.prev_shape = point

    def erase(self, index_tip, middle_tip):
        cx = (index_tip[0] + middle_tip[0]) // 2
        cy = (index_tip[1] + middle_tip[1]) // 2
        self.erase_history.append((cx, cy))
        point = smooth(self.erase_history)
        if point is None:
            return
        self.cursor_point = point
        self.cursor_radius = ERASER_SIZE // 2

        if self.in_ui_region(point):
            self.prev_erase = None
            return

        self.reset_click_flags()

        oid = self.find_shape_at(point)
        if oid is not None:
            self.remove_shape(oid)

        if self.prev_erase is not None:
            cv2.line(self.canvas, self.prev_erase, point, (0, 0, 0), ERASER_SIZE)
            cv2.line(self.alpha, self.prev_erase, point, 0, ERASER_SIZE)
        cv2.circle(self.canvas, point, ERASER_SIZE // 2, (0, 0, 0), -1)
        cv2.circle(self.alpha, point, ERASER_SIZE // 2, 0, -1)
        self.prev_erase = point

    def handle_selection(self, x, y, dist=None):
        if x < TOOLBAR_WIDTH or y < PANEL_HEIGHT:
            self.handle_panel((x, y))
            return
        if self.pinch_active_prev:
            if self.dragging and self.drag_prev is not None:
                dx = x - self.drag_prev[0]
                dy = y - self.drag_prev[1]
                if dx or dy:
                    self.move_selection(dx, dy)
                if dist is not None and self.pinch_dist_prev and self.selected_ids:
                    ratio = dist / self.pinch_dist_prev
                    if 0.2 < ratio < 5 and abs(ratio - 1) > 0.015:
                        self.scale_selection(ratio)
            self.drag_prev = (x, y)
            if dist is not None:
                self.pinch_dist_prev = dist
        else:
            self.pinch_active_prev = True
            self.select_tap((x, y))
            if dist is not None:
                self.pinch_dist_prev = dist

    def in_ui_region(self, point):
        x, y = point
        return x < TOOLBAR_WIDTH or y < PANEL_HEIGHT

    def scale_selection(self, ratio):
        pts = []
        for oid in self.selected_ids:
            obj = self.obj_of(oid)
            if obj:
                pts.append(obj["center"])
        if not pts:
            return
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        moved = set()
        for oid in list(self.selected_ids):
            obj = self.obj_of(oid)
            if obj is None:
                continue
            g = obj["group"]
            members = list(self.groups.get(g, ())) if g is not None else [oid]
            for mid in members:
                if mid in moved:
                    continue
                m = self.obj_of(mid)
                if m:
                    px, py = m["center"]
                    nx = int(cx + (px - cx) * ratio)
                    ny = int(cy + (py - cy) * ratio)
                    m["center"] = (
                        max(TOOLBAR_WIDTH + 5, min(FRAME_WIDTH - 5, nx)),
                        max(PANEL_HEIGHT + 5, min(FRAME_HEIGHT - 5, ny)),
                    )
                    m["size"] = int(max(8, min(80, m["size"] * ratio)))
                    moved.add(mid)

    def select_tap(self, point):
        oid = self.find_shape_at(point)
        if oid is not None:
            if oid not in self.selected_ids:
                members = self.get_group_members(oid) or {oid}
                self.selected_ids.update(members)
            self.dragging = True
            self.drag_prev = point
        else:
            self.selected_ids.clear()
            self.dragging = False
            self.drag_prev = None

    def handle_group_gesture(self, action):
        self._reset_stroke_state()
        self.cursor_point = None
        if self.group_action != action:
            self.group_action = action
            self.group_hold = 0
        self.group_hold += 1
        if self.group_hold >= GROUP_HOLD_FRAMES:
            if action == "group":
                self.make_group()
            else:
                self.ungroup_selected()
            self.group_action = None
            self.group_hold = 0

    def _reset_group_gesture(self):
        self.group_action = None
        self.group_hold = 0

    def find_shape_at(self, point):
        for s in reversed(self.shapes):
            x1, y1, x2, y2 = shape_bbox(s["kind"], s["center"][0], s["center"][1], s["size"])
            if rect_contains((x1, y1, x2, y2), point[0], point[1]):
                return s["id"]
        return None

    def get_group_members(self, oid):
        obj = self.obj_of(oid)
        if obj is None:
            return set()
        g = obj["group"]
        if g is None:
            return set()
        return set(self.groups.get(g, ()))

    def obj_of(self, oid):
        for s in self.shapes:
            if s["id"] == oid:
                return s
        return None

    def move_selection(self, dx, dy):
        moved = set()
        for oid in list(self.selected_ids):
            obj = self.obj_of(oid)
            if obj is None:
                continue
            g = obj["group"]
            members = list(self.groups.get(g, ())) if g is not None else [oid]
            for mid in members:
                if mid in moved:
                    continue
                m = self.obj_of(mid)
                if m:
                    x, y = m["center"]
                    nx = max(TOOLBAR_WIDTH + 5, min(FRAME_WIDTH - 10, x + dx))
                    ny = max(PANEL_HEIGHT + 10, min(FRAME_HEIGHT - 10, y + dy))
                    m["center"] = (nx, ny)
                    moved.add(mid)

    def make_group(self):
        sel = [oid for oid in self.selected_ids if self.obj_of(oid) is not None]
        if len(sel) >= 2:
            gid = self.next_group_id
            self.next_group_id += 1
            self.groups[gid] = set(sel)
            for oid in sel:
                self.obj_of(oid)["group"] = gid
            print(f"Grouped {len(sel)} shapes into group {gid}")

    def ungroup_selected(self):
        gids = set()
        for oid in list(self.selected_ids):
            obj = self.obj_of(oid)
            if obj and obj["group"] is not None:
                gids.add(obj["group"])
        for gid in gids:
            for oid in self.groups.pop(gid, ()):
                obj = self.obj_of(oid)
                if obj:
                    obj["group"] = None
            print(f"Ungrouped group {gid}")

    def remove_shape(self, oid):
        obj = self.obj_of(oid)
        if obj is None:
            return
        g = obj["group"]
        if g is not None:
            members = self.groups.get(g)
            if members:
                members.discard(oid)
                if len(members) <= 1:
                    for mid in members:
                        m = self.obj_of(mid)
                        if m:
                            m["group"] = None
                    self.groups.pop(g, None)
        self.shapes = [s for s in self.shapes if s["id"] != oid]
        self.selected_ids.discard(oid)

    def handle_panel(self, point):
        x, y = point
        for rect in self.shape_slots:
            if rect_contains(rect[:4], x, y):
                self.current_shape = rect[4]
                return
        if y >= PANEL_HEIGHT:
            return
        for name, rect in self.tool_rects.items():
            if rect_contains(rect, x, y):
                if not self.tool_once:
                    self.tool_once = True
                    self.tool = name.lower()
                return
        for rect, color in self.swatches:
            if rect_contains(rect, x, y):
                self.color = color
                return
        for name, rect in self.group_rects.items():
            if rect_contains(rect, x, y):
                if name == "GROUP" and not self.group_once:
                    self.group_once = True
                    self.make_group()
                elif name == "UNGROUP" and not self.ungroup_once:
                    self.ungroup_once = True
                    self.ungroup_selected()
                return
        for name, rect in self.button_rects.items():
            if rect_contains(rect, x, y):
                if name == "SAVE" and not self.saved_once:
                    self.saved_once = True
                    self.save()
                elif name == "CLEAR" and not self.clear_once:
                    self.clear_once = True
                    self.clear_canvas()
                elif name == "EXIT" and not self.exit_once:
                    self.exit_once = True
                    self.exit_requested = True
                return

    def reset_click_flags(self):
        self.saved_once = False
        self.clear_once = False
        self.exit_once = False
        self.group_once = False
        self.ungroup_once = False
        self.tool_once = False

    def clear_canvas(self):
        self.canvas[:] = 0
        self.alpha[:] = 0
        self.shapes.clear()
        self.selected_ids.clear()
        self.groups.clear()

    def paint_color(self):
        if self.color is None:
            c = rainbow_bgr(self.hue)
            self.hue = (self.hue + 6) % 180
            return c
        return self.color

    def render_shapes(self):
        self.shape_layer[:] = 0
        self.shape_alpha[:] = 0
        for s in self.shapes:
            draw_shape(
                self.shape_layer, self.shape_alpha, s["kind"],
                s["center"][0], s["center"][1], s["size"], s["color"],
            )

    def draw_selection_ui(self, img):
        for oid in self.selected_ids:
            obj = self.obj_of(oid)
            if obj is None:
                continue
            x1, y1, x2, y2 = shape_bbox(obj["kind"], obj["center"][0], obj["center"][1], obj["size"])
            color = (0, 255, 255) if obj["group"] is not None else (255, 255, 0)
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 2)
            for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                cv2.circle(img, (hx, hy), 4, color, -1)

    def board_image(self, background):
        img = background.copy()
        mask = self.alpha.astype(bool)
        img[mask] = self.canvas[mask]
        self.render_shapes()
        smask = self.shape_alpha.astype(bool)
        img[smask] = self.shape_layer[smask]
        return img

    def save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        black = np.zeros_like(self.frame)
        canvas_path = os.path.join(OUTPUT_DIR, f"canvas_{stamp}.png")
        cv2.imwrite(canvas_path, self.board_image(black))
        print(f"Saved canvas: {canvas_path}")
        board_path = os.path.join(OUTPUT_DIR, f"board_{stamp}.png")
        cv2.imwrite(board_path, self.board_image(self.frame.copy()))
        print(f"Saved board: {board_path}")

    def compose(self):
        result = self.frame.copy()
        mask = self.alpha.astype(bool)
        result[mask] = self.canvas[mask]

        self.render_shapes()
        smask = self.shape_alpha.astype(bool)
        result[smask] = self.shape_layer[smask]

        self.draw_selection_ui(result)

        if self.cursor_point:
            if self.mode == "erase":
                cv2.circle(result, self.cursor_point, self.cursor_radius, (0, 0, 255), 2)
            elif self.mode == "shape":
                draw_shape(
                    result, None, self.current_shape,
                    self.cursor_point[0], self.cursor_point[1],
                    DEFAULT_SHAPE_SIZE, (255, 255, 255), fill=False,
                    outline=(255, 255, 255), thickness=2,
                )
            else:
                cv2.circle(result, self.cursor_point, self.cursor_radius, self.paint_color(), 2)

        if self.group_action:
            label = "GROUPING..." if self.group_action == "group" else "UNGROUPING..."
            frac = min(1.0, self.group_hold / GROUP_HOLD_FRAMES)
            x1 = FRAME_WIDTH // 2 - 175
            y1 = FRAME_HEIGHT - 90
            overlay = result.copy()
            rounded_rect(overlay, (x1, y1), (x1 + 350, y1 + 64), 14, (15, 15, 20), -1)
            result = cv2.addWeighted(overlay, 0.72, result, 0.28, 0)
            rounded_rect(result, (x1, y1), (x1 + 350, y1 + 64), 14, (255, 255, 255), 1)
            cv2.putText(
                result, label, (x1 + 20, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                result, f"{int(frac * 100)}%", (x1 + 290, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            bx1, by1 = x1 + 20, y1 + 46
            bw, bh = 310, 10
            rounded_rect(result, (bx1, by1), (bx1 + bw, by1 + bh), 5, (80, 80, 90), -1)
            if frac > 0:
                rounded_rect(result, (bx1, by1), (bx1 + int(bw * frac), by1 + bh), 5, (0, 255, 255), -1)
        return result

    def draw_ui(self):
        # ---- left vertical shape toolbar (full height) ----
        cv2.rectangle(self.frame, (0, 0), (TOOLBAR_WIDTH, FRAME_HEIGHT), (38, 38, 46), -1)
        cv2.line(self.frame, (TOOLBAR_WIDTH, 0), (TOOLBAR_WIDTH, FRAME_HEIGHT), (75, 75, 85), 1)
        cv2.putText(self.frame, "SHAPES", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for rect in self.shape_slots:
            x1, y1, x2, y2, kind = rect
            cx = (x1 + x2) // 2
            if kind == self.current_shape:
                rounded_rect(self.frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), 10, (82, 82, 96), -1)
                rounded_rect(self.frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), 10, (0, 255, 255), 2)
            else:
                rounded_rect(self.frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), 10, (50, 50, 60), -1)
                rounded_rect(self.frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), 10, (115, 115, 125), 1)
            draw_shape(self.frame, None, kind, cx, y1 + 26, 15, (235, 235, 235), outline=(255, 255, 255), thickness=2)
            label = kind.upper()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
            cv2.putText(self.frame, label, (cx - tw // 2, y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (195, 195, 205), 1, cv2.LINE_AA)

        # ---- top panel ----
        cv2.rectangle(self.frame, (TOOLBAR_WIDTH, 0), (FRAME_WIDTH, PANEL_HEIGHT), (32, 32, 38), -1)
        cv2.rectangle(self.frame, (0, PANEL_HEIGHT), (FRAME_WIDTH, FRAME_HEIGHT), (190, 190, 190), 1)
        cv2.line(self.frame, (TOOLBAR_WIDTH, 40), (FRAME_WIDTH, 40), (58, 58, 66), 1)

        tool_colors = {"DRAW": (50, 170, 60), "SHAPE": (150, 90, 220), "SELECT": (30, 155, 255)}
        for name, rect in self.tool_rects.items():
            active = self.tool == name.lower()
            color = tool_colors[name] if active else shade(tool_colors[name], 0.5)
            draw_button(self.frame, rect, name, color, active, scale=0.5)

        for rect, color in self.swatches:
            x1, y1, x2, y2 = rect
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = (x2 - x1) // 2
            cv2.circle(self.frame, (cx + 1, cy + 2), r, (12, 12, 16), -1)
            if color is None:
                for i in range(12):
                    a0 = i * 30
                    cv2.ellipse(self.frame, (cx, cy), (r, r), 0, a0, a0 + 31, rainbow_bgr(i * 15), -1)
            else:
                cv2.circle(self.frame, (cx, cy), r, color, -1)
                cv2.circle(self.frame, (cx - r // 3, cy - r // 3), max(2, r // 4), shade(color, 1.5), -1)
            cv2.circle(self.frame, (cx, cy), r, (255, 255, 255), 1)
            if color == self.color:
                cv2.circle(self.frame, (cx, cy), r + 4, (255, 255, 255), 2)
                glow = shade(color, 1.35) if color is not None else (255, 255, 255)
                cv2.circle(self.frame, (cx, cy), r + 8, glow, 1)

        group_colors = {"GROUP": (150, 80, 220), "UNGROUP": (70, 70, 85)}
        for name, rect in self.group_rects.items():
            bright = name == "GROUP" and len(self.selected_ids) >= 2
            color = group_colors[name] if bright else shade(group_colors[name], 0.55)
            draw_button(self.frame, rect, name, color, bright, scale=0.45)

        button_colors = {"SAVE": (45, 180, 55), "CLEAR": (20, 130, 240), "EXIT": (220, 35, 35)}
        for name, rect in self.button_rects.items():
            draw_button(self.frame, rect, name, button_colors[name], False, scale=0.45)

    def to_pixel(self, lm):
        return int(lm.x * FRAME_WIDTH), int(lm.y * FRAME_HEIGHT)

    def _reset_stroke_state(self):
        self.prev_draw = None
        self.prev_erase = None
        self.prev_shape = None
        self.draw_history.clear()
        self.erase_history.clear()

    def _reset_pinch(self):
        self.pinch_active_prev = False
        self.dragging = False
        self.drag_prev = None
        self.pinch_dist_prev = None

    def _reset_tracking(self):
        self._reset_stroke_state()
        self._reset_pinch()
        self._reset_group_gesture()
        self.cursor_point = None


def main():
    board = VirtualWritingBoard()
    board.run()


if __name__ == "__main__":
    main()
