import argparse

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
        print(f"Only in checkpoint A: {encoder1_keys - encoder2_keys}")
        print(f"Only in checkpoint B: {encoder2_keys - encoder1_keys}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Compare two BAKU checkpoints")
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--state-key", default="actor")
    args = parser.parse_args()

    print(f"Loading checkpoint A: {args.checkpoint_a}")
    state_dict_a = torch.load(
        args.checkpoint_a, weights_only=False, map_location="cpu"
    )
    print(f"Loading checkpoint B: {args.checkpoint_b}")
    state_dict_b = torch.load(
        args.checkpoint_b, weights_only=False, map_location="cpu"
    )

    print(f"\nCheckpoint A type: {type(state_dict_a)}")
    print(f"Checkpoint B type: {type(state_dict_b)}")
    if isinstance(state_dict_a, dict) and isinstance(state_dict_b, dict):
        print(f"Checkpoint A keys: {list(state_dict_a.keys())}")
        print(f"Checkpoint B keys: {list(state_dict_b.keys())}")
        encoders_are_different = compare_encoder_weights(
            state_dict_a, state_dict_b, args.state_key
        )
        status = "DIFFERENT" if encoders_are_different else "IDENTICAL"
        print(f"\nEncoder weights are {status} between checkpoints")
    else:
        print("ERROR: One or both files are not dictionaries")

    print("\nStructure of checkpoint A:")
    print("=" * 50)
    if isinstance(state_dict_a, dict):
        print_dict_structure(state_dict_a, max_depth=2)


if __name__ == "__main__":
    main()
