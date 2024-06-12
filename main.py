import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from ccl import ccl, show_labeled_image, labeled_image
from block_matching import blockMatching, preprocess_image
from morphology import morphology_opening, morphology_closing
from otsu import otsu_recursive, binary_threshold, get_histogram
from flocking import calculate_new_motion_vector

def threshold_image(image_array, threshold=250):
    thresholded_array = np.where(image_array >= threshold, 255, 0)
    return thresholded_array.astype(np.uint8)

params = {
    'separation_distance': 5.0,
    'avoidance_distance': 3.0,
    'separation_weight': 1.5,
    'alignment_weight': 1.0,
    'cohesion_weight': 1.0,
    'avoidance_weight': 2.0,
    'randomness_weight': 0.5
}

# colors
colors = np.random.randint(0, 256, size=(10, 3)).tolist()

class Agent:
    def __init__(self, id: int, initial_pos: tuple):
        self.id = id
        self.pos = initial_pos
        self.tail = []
        self.new_pos_x = np.random.randint(-5, 6)
        self.new_pos_y = np.random.randint(-5, 6)
        # self.prev_img = np.zeros((200, 200), dtype=np.uint8)
        
        self.motion_x = 0
        self.motion_y = 0

    def add_to_tail(self, pos):
        self.tail.append(pos)

    def update_direction(self):
        # move randomly
        self.new_pos_x = np.random.randint(-5, 6)
        self.new_pos_y = np.random.randint(-5, 6)

    def move_random(self):
        # boundary check
        if self.pos[0] + self.new_pos_x > 710:
            self.new_pos_x = -3
            
        elif self.pos[0] + self.new_pos_x < 10:
            self.new_pos_x = 3
            
        if self.pos[1] + self.new_pos_y > 1270:
            self.new_pos_y = -3
            
        elif self.pos[1] + self.new_pos_y < 10:
            self.new_pos_y = 3
        
        # update position
        self.pos = (self.pos[0] + self.new_pos_x, self.pos[1] + self.new_pos_y)
        self.add_to_tail(self.pos)
        
    def forward(self):
        self.pos = (int(self.pos[0] + self.new_pos_x), int(self.pos[1] + self.new_pos_y))
        self.add_to_tail(self.pos)
        
    def move(self, prev_img, curr_img):
        prev_img_hist = get_histogram(prev_img)
        curr_img_hist = get_histogram(curr_img)
        
        prev_img_otsu_thres = otsu_recursive(prev_img_hist)
        curr_img_otsu_thres = otsu_recursive(curr_img_hist)
        
        prev_img_binary = binary_threshold(prev_img, prev_img_otsu_thres)
        curr_img_binary = binary_threshold(curr_img, curr_img_otsu_thres)
        
        prev_img_morph_open = morphology_opening(prev_img_binary, 3)
        curr_img_morph_open = morphology_opening(curr_img_binary, 3)
        
        prev_img_ccl = ccl(prev_img_morph_open)
        curr_img_ccl = ccl(curr_img_morph_open)
        
        n_component = min(prev_img_ccl.max(), curr_img_ccl.max())
        motion_vectors = []
        # # print(n_component)
        
        for comp in range(1, n_component + 1):
            if comp == self.id:
                continue
            
            print("* ", end="")
            prev_img_comp = preprocess_image(prev_img_ccl.copy(), comp)
            curr_img_comp = preprocess_image(curr_img_ccl.copy(), comp)
            
            try:
                v = blockMatching(prev_img_comp, curr_img_comp)
                motion_vectors.append(v)
            except:
                v = [0, 0]
                motion_vectors.append(v)
            
        # # print("\n", len(motion_vectors))
        print(motion_vectors)
        
        # # print(motion_vectors)
        # Flocking Algorithm
        # agent_position = np.array(self.pos)
        agent_position = np.array([0, 0])
        agent_motion = np.array([self.motion_x, self.motion_y])
        
        new_motion_vector = calculate_new_motion_vector(motion_vectors, agent_position, agent_motion, params)
        
        self.motion_x = new_motion_vector[0] * 2
        self.motion_y = new_motion_vector[1] * 2
        
        self.new_pos_x = self.motion_x
        self.new_pos_y = self.motion_y
        
        print(new_motion_vector)
        
        self.pos = (int(self.pos[0] + self.motion_x), int(self.pos[1] + self.motion_y))
        
        # 7. Update tail
        self.add_to_tail(self.pos)
        
        return curr_img_ccl
    
        
    def get_tail(self):
        if len(self.tail) < 5:
            return self.tail
        
        return self.tail[-5:]

class MotherShip(Agent):
    def __init__(self, id: int, initial_pos: tuple):
        super().__init__(id, initial_pos)

def update_frame(frame, agents):
    frame = np.zeros((720, 1280), dtype=np.uint8)
    for agent in agents:        
        for i, tail_pos in enumerate(agent.get_tail()):
            frame[tail_pos[0] - 3:tail_pos[0] + 3, tail_pos[1] - 3:tail_pos[1] + 3] = (i + 1) * 40
            
        frame[agent.pos[0] - 3:agent.pos[0] + 3, agent.pos[1] - 3:agent.pos[1] + 3] = 255
        
    return frame

def get_agent_view(agent, frame):
    # return near 10 x 10 view of the agent
    # # print(v.shape)
    return frame[agent.pos[0] - 100:agent.pos[0] + 100, agent.pos[1] - 100:agent.pos[1] + 100]

def main():
    frame = np.zeros((720, 1280), dtype=np.uint8)
    
    agents = [Agent(0, (280, 280)), Agent(1, (280, 300)), Agent(2, (300, 280)), Agent(3, (300, 300)), Agent(4, (310, 310)), 
              Agent(5, (320, 320)), Agent(6, (280, 310)), Agent(7, (310, 280)), Agent(8, (320, 310)), Agent(9, (310, 320)),
              Agent(10, (330, 330)), Agent(11, (330, 340)), Agent(12, (340, 330)), Agent(13, (340, 340)), Agent(14, (350, 350)),
              Agent(15, (360, 360)), Agent(16, (330, 350)), Agent(17, (350, 330)), Agent(18, (360, 350)), Agent(19, (350, 360)),]

    prev_views = []

    frame = update_frame(frame, agents)
    cv.imshow('frame', frame)

    for i in tqdm(range(1000)):
        # for j in range(len(agents)):
                # v = get_agent_view(agent[j], frame)
                # agents[j].move_random()
                # agent[j].move(v)
                
                # if i % 4 == 0:
                #     agents[j].update_direction()
        
        # frame = update_frame(frame, agents)
        
        render = []
        agent_views = []
        for j, agent in enumerate(agents):
            agent_view = frame[agent.pos[0] - 100:agent.pos[0] + 100, agent.pos[1] - 100:agent.pos[1] + 100]
            prev_view = prev_views[j] if len(prev_views) > 0 else agent_view
            agent_views.append(agent_view)

            if j > 100:
                agent.move_random()
                
                # if i % 10 == 0:
                if np.random.randint(0, 10) == 0:
                    agent.update_direction()
                
            else:
                # plt.imshow(agent_view)
                # plt.show()
                # plt.imshow(prev_view)
                # plt.show()
                thres_prev_view = threshold_image(prev_view)
                thres_agent_view = threshold_image(agent_view)
                
                # agent.move(thres_prev_view, thres_agent_view)
                if np.random.randint(0, 5) == 0:
                    ccl = agent.move(thres_prev_view, thres_agent_view)
                    render.append(labeled_image(ccl))
                else:
                    agent.forward()
                    v = cv.cvtColor(agent_view, cv.COLOR_GRAY2RGB)
                    render.append(v)
                
        prev_views = agent_views.copy()
        
        frame = update_frame(frame, agents)
        cv.imshow('frame', frame)
        
        if len(render) > 0:
            concatenated_view1 = np.concatenate(render[:10], axis=1)
            concatenated_view2 = np.concatenate(render[10:], axis=1)
            
            concatenated_view = np.concatenate([concatenated_view1, concatenated_view2])
            
            cv.imshow('agents view (1 ~ 10)', concatenated_view)
        
        
        # cv.imshow('agents view (11 ~ 20)', concatenated_view2)
        
        # prev_concatenated_view = np.concatenate(prev_views, axis=1)
        # cv.imshow('prev agents view', prev_concatenated_view)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
    