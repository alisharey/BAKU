import torch

def print_dict_structure(d, indent=0, max_depth=3):
    """Recursively print dictionary structure with indentation"""
    if indent > max_depth:
        print("  " * indent + "... (max depth reached)")
        return
    
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}: (dict with {len(value)} keys)")
            print_dict_structure(value, indent + 1, max_depth)
        elif isinstance(value, torch.Tensor):
            print("  " * indent + f"{key}: Tensor{tuple(value.shape)} [{value.dtype}]")
        elif isinstance(value, (list, tuple)):
            print("  " * indent + f"{key}: {type(value).__name__} (length: {len(value)})")
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print("  " * (indent + 1) + f"First element: Tensor{tuple(value[0].shape)} [{value[0].dtype}]")
        else:
            print("  " * indent + f"{key}: {type(value).__name__} = {value}")

def compare_encoder_weights(state_dict1, state_dict2, key_path="actor"):
    """Compare encoder weights between two state dictionaries"""
    
    # Navigate to encoder in both dicts
    encoder1 = state_dict1.get(key_path)
    encoder2 = state_dict2.get(key_path)
    
    if encoder1 is None or encoder2 is None:
        print(f"ERROR: {key_path} not found in one or both files")
        return
    
    print(f"\nComparing {key_path} weights:")
    print("=" * 50)
    
    # Get all encoder parameter keys
    encoder1_keys = set(encoder1.keys()) if isinstance(encoder1, dict) else set()
    encoder2_keys = set(encoder2.keys()) if isinstance(encoder2, dict) else set()
    
    if encoder1_keys != encoder2_keys:
        print(f"WARNING: Different keys in encoders!")
        print(f"Only in 50000: {encoder1_keys - encoder2_keys}")
        print(f"Only in 30000: {encoder2_keys - encoder1_keys}")
    
    # Compare each parameter
    common_keys = encoder1_keys & encoder2_keys
    different_params = 0
    identical_params = 0
    
    for param_key in sorted(common_keys):
        param1 = encoder1[param_key]
        param2 = encoder2[param_key]
        
        if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
            if param1.shape != param2.shape:
                print(f"  {param_key}: SHAPE MISMATCH - {param1.shape} vs {param2.shape}")
                different_params += 1
            else:
                # Check if tensors are identical
                are_identical = torch.equal(param1, param2)
                
                if are_identical:
                    print(f"  {param_key}: IDENTICAL")
                    identical_params += 1
                else:
                    # Calculate difference statistics
                    diff = torch.abs(param1 - param2)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    
                    print(f"  {param_key}: DIFFERENT - max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}")
                    different_params += 1
        else:
            print(f"  {param_key}: Non-tensor comparison - {type(param1)} vs {type(param2)}")
    
    print(f"\nSummary:")
    print(f"  Identical parameters: {identical_params}")
    print(f"  Different parameters: {different_params}")
    print(f"  Total compared: {len(common_keys)}")
    
    return different_params > 0

# Load both checkpoint files
weights_path_50k = "/l/users/ali.abouzeid/BAKU/baku/exp_local/2025.07.18_train/deterministic/230614_vggt/snapshot/50000.pt"
weights_path_30k = "/l/users/ali.abouzeid/BAKU/baku/exp_local/2025.07.18_train/deterministic/230614_vggt/snapshot/0.pt"

print("Loading 50k checkpoint...")
state_dict_50k = torch.load(weights_path_50k, weights_only=False, map_location='cpu')

print("Loading 30k checkpoint...")
state_dict_30k = torch.load(weights_path_30k, weights_only=False, map_location='cpu')

print(f"\n50k file type: {type(state_dict_50k)}")
print(f"30k file type: {type(state_dict_30k)}")

if isinstance(state_dict_50k, dict) and isinstance(state_dict_30k, dict):
    print(f"50k keys: {list(state_dict_50k.keys())}")
    print(f"30k keys: {list(state_dict_30k.keys())}")
    
    # Compare encoders
    encoders_are_different = compare_encoder_weights(state_dict_50k, state_dict_30k, "actor")
    
    if encoders_are_different:
        print(f"\n✅ CONFIRMED: Encoder weights are DIFFERENT between checkpoints")
    else:
        print(f"\n❌ WARNING: Encoder weights are IDENTICAL between checkpoints")
        
else:
    print("ERROR: One or both files are not dictionaries")

# Optional: Print structure of first file
print(f"\nStructure of 50k checkpoint:")
print("=" * 50)
if isinstance(state_dict_50k, dict):
    print_dict_structure(state_dict_50k, max_depth=2)