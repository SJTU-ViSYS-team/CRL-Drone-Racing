import json
import random
import os

def regenerate_obstacles(file_path, num_obstacles=18):
    # Read the existing JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Get existing object instances
    object_instances = data["object_instances"]

    # Remove existing obstacles
    # object_instances = [obj for obj in object_instances if obj["template_name"] != "0.1_0.1_2cylinder"]
    object_instances = [obj for obj in object_instances if obj["template_name"] != "0.15_0.15tree"]
    
    # ######################### demo1 ############################
    
    # # # Define exclusion points to avoid
    # exclusion_points = [(-1, 0, 0),(-5, 0, 0),(-9, 0, 0),(-12, 0, 0)]
    
    # # Generate obstacle region 1 (y-range, fixed z, x-range)
    # for _ in range(int(num_obstacles/3)):
    #     while True:
    #         y = random.uniform(-2., 2.)
    #         x = random.uniform(-6.5, -3.5)
    #         z = 0.0
    #         # Check whether it is close to any exclusion point
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:  # distance squared < 0.5^2
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [y, z, x],  # original coordinate order: [y, z, x]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # # Generate obstacle region 2 (x-range, fixed y, z-range)
    # for _ in range(int(num_obstacles/3)):
    #     while True:
    #         x = random.uniform(-2., 2.)
    #         z = random.uniform(-10.5, -7.5)
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # # Generate obstacle region 3 (x-range, fixed y, z-range)
    # for _ in range(int(num_obstacles/3)):
    #     while True:
    #         x = random.uniform(-2., 2.)
    #         z = random.uniform(-14.5, -11.5)
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)

############################# demo2 #####################################
    # Randomly generate obstacle positions and rotations
    exclusion_points = [(-2, -3, 0),(-6, 3, 0),(-7, -2, 0),(-1, 2, 0)]
    
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(-3., -1.)  # x coordinate
            z = random.uniform(-7.5, -4.5)  # z coordinate
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # original coordinate order: [x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(1., 3.) # x coordinate
            z = random.uniform(-7.5, -4.5)  # z coordinate
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # original coordinate order: [x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(1., 3.)  # x coordinate
            z = random.uniform(-3.5, -0.5)   # z coordinate
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # original coordinate order: [x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(-3., -1.)   # x coordinate
            z = random.uniform(-3.5, -0.5)   # z coordinate
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # original coordinate order: [x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
############################################################################################################
    #         # Randomly generate obstacle positions and rotations (demo3)
    # exclusion_points = [(-2, -3, 0),(-10, 3, 0),(-10, -3, 0),(-2, 3, 0)]
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(-3.5, -1.)   # x coordinate
    #         z = random.uniform(-3.5, -1.)   # z coordinate
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(-3.5, -1.)   # x coordinate
    #         z = random.uniform(-7.5, -4.5)   # z coordinate
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(-3.5, -1.)   # x coordinate
    #         z = random.uniform(-11., -8.5)   # z coordinate
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.1_0.1_2cylinder",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(1, 3.5)   # x coordinate
    #         z = random.uniform(-3.5, -1.)   # z coordinate
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.1_0.1_2cylinder",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(1, 3.5)   # x coordinate
    #         z = random.uniform(-7.5, -4.5)   # z coordinate
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(1, 3.5)   # x coordinate
    #         z = random.uniform(-11., -8.5)   # z coordinate
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # original coordinate order: [x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
############################################################################################################
        
    # Update JSON data
    data["object_instances"] = object_instances

    return data


def generate_random_json_files(base_file_path, output_dir, num_files=4):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read base JSON file
    base_data = regenerate_obstacles(base_file_path)

    for i in range(num_files):
        # Generate a different obstacle count for each file
        # num_obstacles = random.randint(5, 10)  # randomly select obstacle count
        num_obstacles = 16
        modified_data = regenerate_obstacles(base_file_path, num_obstacles)

        # Generate random filename
        random_filename = f"racing_{i+1}.scene_instance.json"
        output_file_path = os.path.join(output_dir, random_filename)

        # Write updated JSON data to new file
        with open(output_file_path, 'w') as file:
            json.dump(modified_data, file, indent=4)

        print(f"File '{random_filename}' generated successfully in '{output_dir}'.")

# example usage

# base_file_path = 'datasets/spy_datasets/configs/racing/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/racing'
# base_file_path = 'datasets/spy_datasets/configs/racing8_random_ob/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/racing8_random_ob'
# base_file_path = 'datasets/spy_datasets/configs/racing_straight/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/racing_straight'
# base_file_path = 'datasets/spy_datasets/configs/demo1_straight_ob1/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo1_straight_random_ob'
base_file_path = 'datasets/spy_datasets/configs/demo2_3Dcircle_ob1/racing_1.scene_instance.json'
output_directory = 'datasets/spy_datasets/configs/demo2_3Dcircle_random_ob'
# base_file_path = 'datasets/spy_datasets/configs/demo1_songjiang/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo1_songjiang'
# base_file_path = 'datasets/spy_datasets/configs/demo2_songjiang/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo2_songjiang'
# base_file_path = 'datasets/spy_datasets/configs/demo3_songjiang/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo3_songjiang'
generate_random_json_files(base_file_path, output_directory, num_files=5)
