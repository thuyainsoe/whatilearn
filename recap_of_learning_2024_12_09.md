# Recap of Learning - Date: 2024-12-09

## 1. Key Libraries & Tools

Today, you explored the essential libraries and tools used in machine learning (ML) and deep learning (DL).
These libraries provide the foundation for building, training, and evaluating machine learning models.

- **TensorFlow (Google)**: A popular library for machine learning and deep learning.
- **PyTorch**: An open-source machine learning framework that emphasizes dynamic computation graphs.
- **NumPy**: A library for numerical computations in Python.
- **Scikit-learn**: A versatile library for machine learning, including classification, regression, and clustering.

---

## 2. Key Concepts in Machine Learning (ML)

Machine learning (ML) involves training models to recognize patterns in data.

- **Classification Models**: Used to categorize data into distinct classes.
- **Regression Models**: Used to predict continuous, numerical values.
- **Clustering Models**: Used to group similar data points together.
- **Dimensionality Reduction**: Reduces the number of input features while preserving essential information.

### Learning Methods

- **Machine Learning (ML)**: Systems that learn from data.
- **Deep Learning (DL)**: Uses neural networks with multiple layers.
- **Artificial Intelligence (AI)**: The broader field of building systems that simulate human intelligence.

---

## 3. Core TensorFlow Functions & Concepts

Tensors are the fundamental building blocks of machine learning models. They are multi-dimensional arrays.

### Operations

- **`tf.tensor(data, shape)`**: Creates a tensor.
- **`tf.add(a, b)`**: Adds two tensors.
- **`tf.multiply(a, b)`**: Multiplies tensors.
- **`tf.slice(tensor, start, size)`**: Extracts a portion of a tensor.

---

## 4. Building a Simple Neural Network

You also learned to build a simple model using TensorFlow's Sequential API.

### Steps to Build a Neural Network

1. **Define the Model**: Use `tf.sequential()` to create a stack of layers.
2. **Add Layers**: Use `tf.layers.dense()` to add dense (fully connected) layers.
3. **Compile the Model**: Set the optimizer and loss function using `model.compile()`.
4. **Train the Model**: Fit the model to the data using `model.fit()`.

### Example Code

```python
model = tf.sequential()
model.add(tf.layers.dense({units: 1, inputShape: [1]}))
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})
model.fit(xs, ys, {epochs: 50})
```

This simple process of defining, compiling, and training a model forms the foundation of most machine learning models.

---

## 5. Projects to Build

These projects will enhance your skills in machine learning and development:

- **PoseNet**: For pose detection.
- **MobileNet**: For image classification.
- **FaceMesh**: For facial landmark detection.
- **Canvas Trial**: Experiments with web canvas and machine learning models.

---

## 6. Vocabulary You Learned

| **Word**           | **Meaning / Usage**                                  |
| ------------------ | ---------------------------------------------------- |
| **Prerequisites**  | Things you must know before starting.                |
| **Leverage**       | To use something to your advantage.                  |
| **Prior**          | Something that happens before.                       |
| **Let's go ahead** | Phrase used to continue or start something.          |
| **Prior to**       | Before a specific event (e.g., "Prior to the exam"). |
| **Prior meeting**  | A meeting that happened earlier.                     |

These vocabulary words are essential for technical discussions, meetings, and programming documentation.

---

## 7. What's Next?

Reinforce your learning by:

- **Building small projects** using PoseNet, MobileNet, or FaceMesh.
- **Practice coding daily** with TensorFlow.
- **Revisit core concepts** like tensors, models, and the role of operations like `tf.add()` and `tf.slice()`.

Building real-world projects will solidify your understanding and prepare you for larger, more complex tasks in the future.

---
