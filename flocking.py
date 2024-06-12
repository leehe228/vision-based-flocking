import numpy as np

def calculate_new_motion_vector(motion_vectors, agent_position, agent_motion, params):
    # Initialize vectors
    separation_vector = np.zeros(2)
    alignment_vector = np.zeros(2)
    cohesion_vector = np.zeros(2)
    avoidance_vector = np.zeros(2)
    randomness_vector = np.random.uniform(-1, 1, 2)
    
    # Parameters for each behavior
    separation_distance = params.get('separation_distance', 1.0)
    avoidance_distance = params.get('avoidance_distance', 1.0)
    separation_weight = params.get('separation_weight', 1.0)
    alignment_weight = params.get('alignment_weight', 1.0)
    cohesion_weight = params.get('cohesion_weight', 1.0)
    avoidance_weight = params.get('avoidance_weight', 1.0)
    randomness_weight = params.get('randomness_weight', 1.0)
    
    num_neighbors = len(motion_vectors)
    
    if num_neighbors == 0:
        return agent_motion
    
    for motion_vector in motion_vectors:
        distance = np.linalg.norm(motion_vector - agent_position)
        
        if distance < separation_distance:
            separation_vector -= (motion_vector - agent_position)
        
        if distance < avoidance_distance:
            avoidance_vector -= (motion_vector - agent_position)
        
        alignment_vector += motion_vector
        cohesion_vector += motion_vector
    
    # Average the alignment and cohesion vectors
    alignment_vector /= num_neighbors
    cohesion_vector /= num_neighbors
    cohesion_vector = cohesion_vector - agent_position
    
    # Normalize the randomness vector
    randomness_vector = randomness_vector / np.linalg.norm(randomness_vector)
    
    # Calculate the new motion vector
    new_motion_vector = (
        separation_weight * separation_vector +
        alignment_weight * alignment_vector +
        cohesion_weight * cohesion_vector +
        avoidance_weight * avoidance_vector +
        randomness_weight * randomness_vector
    )
    
    # Normalize the new motion vector to maintain consistent speed
    new_motion_vector = new_motion_vector / np.linalg.norm(new_motion_vector)
    
    return new_motion_vector