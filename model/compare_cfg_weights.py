import numpy as np

def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        lines = f.read().strip().split('\n')
    
    layers = []
    for line in lines:
        if line.startswith('['):
            layer_type = line.strip('[]')
            layers.append({'type': layer_type})
        elif '=' in line:
            key, value = line.split('=')
            layers[-1][key.strip()] = value.strip()
    return layers

def compare_cfg_weights(cfg_layers, weights):
    ptr = 0
    for layer in cfg_layers:
        if layer['type'] == 'convolutional':
            # Check input/output channels and calculate expected weights
            input_channels = int(layer.get('channels', 0))
            output_channels = int(layer.get('filters', 0))
            kernel_size = int(layer.get('size', 0)) ** 2
            # Calculate expected parameters for Conv2d layer
            expected_params = (input_channels * kernel_size + 1) * output_channels
            
            if ptr + expected_params > len(weights):
                print(f"Layer {layer} has an expected parameter count of {expected_params}, but exceeds available weights.")
            else:
                print(f"Layer {layer['type']} has {expected_params} parameters, weights available: {len(weights[ptr:ptr + expected_params])}")
            ptr += expected_params
        # Handle other layer types if necessary

cfg_layers = parse_cfg('config/yolov3-KiTTi.cfg')
weights_path = 'darknet53.conv.74'
weights = np.fromfile(weights_path, dtype=np.float32)

compare_cfg_weights(cfg_layers, weights)
