import torch

# Load the model
def analyze_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    print(model)
    return model

if __name__ == "__main__":
    analyze_model("sigmoid.pt")
