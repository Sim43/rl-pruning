import torch


def continue_forward_pass(cnn_model, intermediate_output, start_layer, end_layer = "fc4", eval = True):
        """
        Continue the forward pass from the intermediate output to the final output.

        Args:
            intermediate_output (torch.Tensor): Output from the intermediate layer.
            start_layer (str): Name of the layer to start the forward pass from.
            end_layer (str): Name of the layer to return the output from. Default is fc4 if you want the final output.
            eval (bool): Set Model to eval mode if True
        Returns:
            torch.Tensor: output from end layer.
        """
        # Get the layers after the intermediate layer
        layers = []
        capture = False
        for name, module in cnn_model.named_modules():
            if name == start_layer:
                capture = True
            if capture:
                layers.append(module)
                if name == end_layer:
                    break

        if eval:
            cnn_model.eval()

            with torch.no_grad():
                # Pass the intermediate output through the remaining layers
                for layer in layers:
                    intermediate_output = layer(intermediate_output)
        else:
            cnn_model.train()
            for layer in layers:
                
                intermediate_output = layer(intermediate_output)
        return intermediate_output