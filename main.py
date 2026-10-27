import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import heapq
import math
import time

# ================= AI + CAMERA =================

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Camera not detected")

# ================= MAP =================

CELL, ROWS, COLS = 40, 12, 15
START, GOAL = (10, 1), (1, 13)

fixed_obstacles = {
    (4,5),(5,5),(6,5),
    (7,8),(7,9),(7,10),
    (3,11),(4,11),(5,11)
}

grid = [[0]*COLS for _ in range(ROWS)]
for r,c in fixed_obstacles:
    grid[r][c] = 1

# ================= ROBOT =================

robot_x = START[1]*CELL + CELL//2
robot_y = START[0]*CELL + CELL//2

path = []
path_index = 0
robot_running = True
goal_reached = False

virtual_obstacles = set()
last_replan = 0

REPLAN_DELAY = 1.5
DETECTION_LIMIT = 3
recent_detections = []

detection_count = 0
replan_count = 0
path_length = 0
start_time = time.time()

# ================= A* =================

def heuristic(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(p):
    r,c = p
    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    return [
        (r+dr,c+dc)
        for dr,dc in moves
        if 0 <= r+dr < ROWS
        and 0 <= c+dc < COLS
        and grid[r+dr][c+dc] == 0
    ]

def a_star(start,goal):

    q = [(0,start)]
    came = {}
    cost = {start:0}

    while q:
        _,cur = heapq.heappop(q)

        if cur == goal:
            result = [cur]
            while cur in came:
                cur = came[cur]
                result.append(cur)
            return result[::-1]

        for nxt in neighbors(cur):
            new_cost = cost[cur] + 1

            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + heuristic(nxt,goal)
                heapq.heappush(q,(priority,nxt))
                came[nxt] = cur

    return []

path = a_star(START,GOAL)

# ================= OBSTACLE CONTROL =================

def clear_virtual():

    for r,c in virtual_obstacles:
        if (r,c) not in fixed_obstacles:
            grid[r][c] = 0

    virtual_obstacles.clear()

def add_obstacle(row,col,direction):

    clear_virtual()

    # Place obstacle only around the robot's forward area
    if direction == "LEFT":
        cells = [(row-1,col-1),(row,col-1)]
    elif direction == "RIGHT":
        cells = [(row-1,col+1),(row,col+1)]
    else:
        cells = [(row-1,col)]

    for r,c in cells:
        if (
            0 <= r < ROWS and
            0 <= c < COLS
            and (r,c) != GOAL
            and (r,c) not in fixed_obstacles
        ):
            grid[r][c] = 1
            virtual_obstacles.add((r,c))

# ================= WINDOW =================

root = tk.Tk()
root.title("AI Vision Guided Autonomous Robot")
root.geometry("1280x740")

tk.Label(
    root,
    text="AI VISION GUIDED AUTONOMOUS ROBOT",
    font=("Arial",20,"bold")
).pack(pady=7)

tk.Label(
    root,
    text="YOLO + Dynamic A* Autonomous Navigation",
    font=("Arial",11)
).pack()

main = tk.Frame(root)
main.pack(pady=10)

# ================= CAMERA =================

camera_frame = tk.LabelFrame(
    main,text="LIVE AI CAMERA",
    font=("Arial",13,"bold")
)
camera_frame.grid(row=0,column=0,padx=8)

camera_label = tk.Label(camera_frame)
camera_label.pack(padx=8,pady=8)

# ================= MAP =================

map_frame = tk.LabelFrame(
    main,text="DYNAMIC A* NAVIGATION",
    font=("Arial",13,"bold")
)
map_frame.grid(row=0,column=1,padx=8)

canvas = tk.Canvas(
    map_frame,width=600,height=480,bg="white"
)
canvas.pack()

# ================= STATUS =================

status_frame = tk.LabelFrame(
    main,text="SMART ROBOT STATUS",
    font=("Arial",13,"bold")
)
status_frame.grid(row=0,column=2,padx=8)

def label(text,size=11):
    x = tk.Label(
        status_frame,
        text=text,
        font=("Arial",size,"bold" if size>=14 else "normal")
    )
    x.pack(anchor="w",padx=15,pady=5)
    return x

safety_label = label("● SAFE",15)
object_label = label("Object: None")
confidence_label = label("Confidence: --")
position_label = label("Position: None")
distance_label = label("Safety Distance: --")
decision_label = label("Decision: PATH FOLLOWING")
robot_status_label = label("Robot: RUNNING")
astar_label = label("A*: ACTIVE")
direction_label = label("Direction: UP")
detection_label = label("Detections: 0")
replan_label = label("Re-plans: 0")
path_label = label("Path Length: 0")
time_label = label("Time: 0 s")
goal_label = label("Goal: NOT REACHED")

# ================= DRAW MAP =================

def draw_map():

    canvas.delete("all")

    # Grid + obstacles
    for r in range(ROWS):
        for c in range(COLS):

            x1,y1 = c*CELL,r*CELL
            x2,y2 = x1+CELL,y1+CELL

            canvas.create_rectangle(
                x1,y1,x2,y2,
                outline="lightgray"
            )

            if grid[r][c]:
                canvas.create_rectangle(
                    x1,y1,x2,y2,
                    fill="gray",
                    outline="black"
                )

    # A* path
    for r,c in path:
        canvas.create_oval(
            c*CELL+17,r*CELL+17,
            c*CELL+23,r*CELL+23,
            fill="blue"
        )

    # Goal
    gx,gy = GOAL[1]*CELL,GOAL[0]*CELL

    canvas.create_oval(
        gx+8,gy+8,gx+32,gy+32,
        outline="green",width=3
    )

    canvas.create_text(
        gx+20,gy-8,
        text="GOAL",
        font=("Arial",9,"bold")
    )

    # ================= REALISTIC ROBOT =================

    cx,cy = robot_x,robot_y

    # Safety zone
    canvas.create_oval(
        cx-28,cy-28,
        cx+28,cy+28,
        outline="orange",
        dash=(4,3)
    )

    # Wheels
    canvas.create_rectangle(
        cx-18,cy-12,cx-9,cy+13,
        fill="black"
    )

    canvas.create_rectangle(
        cx+9,cy-12,cx+18,cy+13,
        fill="black"
    )

    # Body
    canvas.create_rectangle(
        cx-14,cy-14,cx+14,cy+14,
        fill="gray",
        outline="black",
        width=2
    )

    # Head
    canvas.create_rectangle(
        cx-11,cy-25,cx+11,cy-10,
        fill="lightgray",
        outline="black",
        width=2
    )

    # Antenna
    canvas.create_line(
        cx,cy-25,cx,cy-32,
        width=2
    )

    canvas.create_oval(
        cx-3,cy-35,cx+3,cy-29,
        fill="black"
    )

    # Eyes
    canvas.create_oval(
        cx-7,cy-22,cx-2,cy-17,
        fill="white"
    )

    canvas.create_oval(
        cx+2,cy-22,cx+7,cy-17,
        fill="white"
    )

    # Front sensor
    canvas.create_oval(
        cx-4,cy-5,cx+4,cy+3,
        fill="black"
    )

# ================= CAMERA + YOLO =================

def update_camera():

    global path,path_index
    global robot_running
    global detection_count,replan_count
    global last_replan
    global recent_detections

    ret,frame = cap.read()

    if not ret:
        root.after(100,update_camera)
        return

    results = model(frame,verbose=False)

    found = False
    name = "None"
    conf = 0
    direction = "None"

    # Detect only the most confident object
    best = None

    for box in results[0].boxes:

        c = float(box.conf[0])

        if c >= 0.55 and (best is None or c > best[0]):
            best = (c,box)

    if best:

        conf,box = best
        cls = int(box.cls[0])
        name = model.names[cls]

        x1,y1,x2,y2 = map(int,box.xyxy[0])
        center = (x1+x2)//2

        if center < frame.shape[1]/3:
            direction = "LEFT"
        elif center < frame.shape[1]*2/3:
            direction = "CENTER"
        else:
            direction = "RIGHT"

        found = True

    # ================= DETECTION =================

    if found:

        # Require repeated detection to reduce false movement
        recent_detections.append((name,direction))
        recent_detections = recent_detections[-DETECTION_LIMIT:]

        stable = (
            len(recent_detections) == DETECTION_LIMIT
            and len(set(recent_detections)) == 1
        )

        object_label.config(text="Object: "+name)
        confidence_label.config(
            text=f"Confidence: {conf*100:.1f}%"
        )
        position_label.config(
            text="Position: "+direction
        )

        distance_label.config(
            text="Safety Distance: CLOSE"
        )

        safety_label.config(
            text="● OBSTACLE DETECTED"
        )

        detection_count += 1

        if stable and time.time()-last_replan > REPLAN_DELAY:

            rr = int(robot_y//CELL)
            cc = int(robot_x//CELL)

            add_obstacle(rr,cc,direction)

            new_path = a_star((rr,cc),GOAL)

            if new_path:

                path = new_path
                path_index = 0
                replan_count += 1
                last_replan = time.time()

                robot_running = True

                decision_label.config(
                    text="Decision: SAFE DETOUR"
                )
                robot_status_label.config(
                    text="Robot: AUTO DETOUR"
                )
                astar_label.config(
                    text="A*: RE-PLANNING"
                )

            else:

                robot_running = False

                safety_label.config(
                    text="● EMERGENCY STOP"
                )
                robot_status_label.config(
                    text="Robot: STOPPED"
                )
                decision_label.config(
                    text="Decision: NO SAFE PATH"
                )

    else:

        recent_detections.clear()

        object_label.config(text="Object: None")
        confidence_label.config(text="Confidence: --")
        position_label.config(text="Position: None")
        distance_label.config(text="Safety Distance: SAFE")
        safety_label.config(text="● SAFE")

        if virtual_obstacles:

            clear_virtual()

            current = (
                int(robot_y//CELL),
                int(robot_x//CELL)
            )

            new_path = a_star(current,GOAL)

            if new_path:
                path = new_path
                path_index = 0
                robot_running = True

                decision_label.config(
                    text="Decision: PATH FOLLOWING"
                )
                robot_status_label.config(
                    text="Robot: RUNNING"
                )
                astar_label.config(
                    text="A*: ACTIVE"
                )

    detection_label.config(
        text=f"Detections: {detection_count}"
    )

    replan_label.config(
        text=f"Re-plans: {replan_count}"
    )

    path_label.config(
        text=f"Path Length: {len(path)}"
    )

    time_label.config(
        text=f"Time: {int(time.time()-start_time)} s"
    )

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        "AI VISION ACTIVE",
        (15,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        .7,(255,255,255),2
    )

    rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(rgb).resize(
        (500,350)
    )

    photo = ImageTk.PhotoImage(img)
    camera_label.config(image=photo)
    camera_label.image = photo

    root.after(100,update_camera)

# ================= ROBOT MOVEMENT =================

def move_robot():

    global robot_x,robot_y
    global path_index,robot_running
    global goal_reached,path_length

    if not robot_running or goal_reached:
        root.after(50,move_robot)
        return

    if not path:
        robot_running = False
        root.after(50,move_robot)
        return

    if path_index >= len(path):

        robot_running = False
        goal_reached = True

        robot_status_label.config(
            text="Robot: GOAL REACHED"
        )

        decision_label.config(
            text="Decision: GOAL REACHED"
        )

        goal_label.config(
            text="Goal: REACHED"
        )

        astar_label.config(
            text="A*: COMPLETE"
        )

        return

    r,c = path[path_index]

    tx = c*CELL + CELL//2
    ty = r*CELL + CELL//2

    dx,dy = tx-robot_x,ty-robot_y
    dist = math.hypot(dx,dy)

    # Slow down near target
    speed = 1.3 if dist < 20 else 2

    if dist > 2:

        robot_x += speed*dx/dist
        robot_y += speed*dy/dist

        direction_label.config(
            text="Direction: " +
            ("UP" if abs(dy)>abs(dx) and dy<0 else
             "DOWN" if abs(dy)>abs(dx) else
             "LEFT" if dx<0 else "RIGHT")
        )

    else:

        robot_x,robot_y = tx,ty
        path_index += 1
        path_length += 1

    draw_map()
    root.after(50,move_robot)

# ================= EXIT =================

def close_program():

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

ttk.Button(
    root,
    text="EXIT",
    command=close_program
).pack(pady=8)

# ================= START =================

draw_map()
update_camera()
move_robot()

root.mainloop()
