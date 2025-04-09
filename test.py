import numpy as np
from neural_network import NeuralNetwork

# Create sample data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train neural network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Save the trained weights
np.savez('model_weights.npz', 
         W1=nn.W1, 
         W2=nn.W2, 
         b1=nn.b1, 
         b2=nn.b2)

def get_user_input():
    try:
        print("\nEnter two binary values (0 or 1)")
        x1 = int(input("Enter first value (0 or 1): "))
        x2 = int(input("Enter second value (0 or 1): "))
        
        if x1 not in [0, 1] or x2 not in [0, 1]:
            raise ValueError("Input must be either 0 or 1")
            
        return np.array([[x1, x2]])
    except ValueError as e:
        print(f"Error: {e}")
        return None

def predict():
    while True:
        user_input = get_user_input()
        if user_input is not None:
            prediction = nn.forward(user_input)
            rounded_prediction = np.round(prediction)
            print(f"\nPrediction for input {user_input[0]}: {rounded_prediction[0][0]}")
        
        again = input("\nWould you like to try another prediction? (y/n): ")
        if again.lower() != 'y':
            break

if __name__ == "__main__":
    # Calculate accuracy on training data
    predictions = nn.forward(X)
    rounded_predictions = np.round(predictions)
    accuracy = np.mean(rounded_predictions == y)
    
    print("Neural Network trained on XOR problem")
    print("Model accuracy: {:.2f}%".format(accuracy * 100))
    predict()