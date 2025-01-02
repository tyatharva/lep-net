import os
import shutil

def reorganize_dataset(current_dataset_path, target_dataset_path):
    # Define the directories to process
    sets = ['train', 'val', 'test']
    
    for set_name in sets:
        # Create new A and B directories in the target location
        setA_path = os.path.join(target_dataset_path, f'{set_name}A')
        setB_path = os.path.join(target_dataset_path, f'{set_name}B')
        os.makedirs(setA_path, exist_ok=True)
        os.makedirs(setB_path, exist_ok=True)
        
        # Get all input and target files in the current set
        input_files = sorted([f for f in os.listdir(os.path.join(current_dataset_path, set_name)) if '_input.zarr' in f])
        target_files = sorted([f for f in os.listdir(os.path.join(current_dataset_path, set_name)) if '_target.zarr' in f])
        
        # Move files to the new structure
        for input_file, target_file in zip(input_files, target_files):
            # Source paths
            input_src = os.path.join(current_dataset_path, set_name, input_file)
            target_src = os.path.join(current_dataset_path, set_name, target_file)
            
            # Destination paths
            input_dst = os.path.join(setA_path, input_file)
            target_dst = os.path.join(setB_path, target_file)
            
            # Move the files
            shutil.move(input_src, input_dst)
            shutil.move(target_src, target_dst)
        
        print(f'{set_name} set reorganized successfully.')

# Example usage:
current_dataset_path = './dataset'
target_dataset_path = './pix2pix/LES_dataset'
reorganize_dataset(current_dataset_path, target_dataset_path)

