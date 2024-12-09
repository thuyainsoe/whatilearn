# Recap of Learning - Date: 2024-12-09

## 1. Key Libraries & Tools

Today, you explored the essential libraries and tools used in machine learning (ML) and deep learning (DL).
These libraries provide the foundation for building, training, and evaluating machine learning models.

- **TensorFlow (Google)**: One of the most popular libraries for machine learning and deep learning. It allows you to create, train, and deploy models at scale. TensorFlow also enables you to work with tensors, which are the core data structures in machine learning.
- **PyTorch**: An open-source machine learning framework that emphasizes dynamic computation graphs, making it intuitive to debug and visualize. PyTorch is widely used in research and production.
- **NumPy**: A library for numerical computations in Python. It provides essential support for arrays, matrices, and mathematical operations required for machine learning.
- **Scikit-learn**: A versatile library for machine learning, including simple models for classification, regression, clustering, and dimensionality reduction. It also provides preprocessing tools for data preparation.

These tools are essential for modern AI, ML, and DL workflows, forming the core of any data science or machine learning project.

---

## 2. Key Concepts in Machine Learning (ML)

Machine learning (ML) involves training models to recognize patterns in data. The main concepts you learned today include the following types of ML models and approaches.

### **Types of Models**

- **Classification Models**: Used to categorize data into distinct classes. Example: Classifying emails as "Spam" or "Not Spam."
- **Regression Models**: Used to predict continuous, numerical values. Example: Predicting house prices based on features like square footage and location.
- **Clustering Models**: Used to group similar data points together. Example: Customer segmentation, where customers are grouped based on purchasing habits.
- **Dimensionality Reduction**: Reduces the number of input features (or dimensions) while preserving essential information. This makes data more manageable and speeds up training.

### **Learning Methods**

- **Machine Learning (ML)**: Systems that learn from data, identifying patterns and making predictions.
- **Deep Learning (DL)**: A subset of ML that uses neural networks with multiple layers (deep architectures) to process large, complex datasets.
- **Artificial Intelligence (AI)**: The broader field of building systems that can simulate human intelligence.

With these concepts, you have a clearer picture of how AI, ML, and DL relate to each other. You also now understand which models to apply to specific problem types (like classification, regression, and clustering).

---

## 3. Core TensorFlow Functions & Concepts

Since TensorFlow is one of the most widely used frameworks for ML and DL, you dove into its most crucial concepts and functions. Here’s a breakdown of what you learned.

### **Tensors**

Tensors are the fundamental building blocks of machine learning models. They are multi-dimensional arrays (like NumPy arrays) that allow machine learning algorithms to perform computations. Tensors can be 0D, 1D, 2D, 3D, or even higher-dimensional.

- **0D Tensor (Scalar)**: A single number (e.g., `4`)
- **1D Tensor (Vector)**: A list of numbers (e.g., `[1, 2, 3]`)
- **2D Tensor (Matrix)**: A table of numbers (e.g., a 2x2 grid `[[1, 2], [3, 4]]`)
- **Higher-Dimensional Tensors**: More complex data structures often used in deep learning, like images (3D) or video frames (4D).

### **Tensor Operations**

To manipulate tensors, you can perform mathematical operations. Here are some of the most critical TensorFlow operations you practiced:

- **`tf.tensor(data, shape)`**: Creates a tensor with specific data and shape.
- **`tf.add(a, b)`**: Adds two tensors element-wise.
- **`tf.multiply(a, b)`**: Multiplies tensors element-wise.
- **`tf.slice(tensor, start, size)`**: Extracts a specific portion (or "slice") of a tensor.

These operations are the basis for neural network computations, where weights, biases, and activations are updated during training.

---

## 4. Building a Simple Neural Network

Today, you also learned to build a basic model using TensorFlow's Sequential API. Here is a step-by-step process of how it works.

### **Step 1: Define the Model**

Use `tf.sequential()` to create a simple stack of layers.

```python
model = tf.sequential()
```

### **Step 2: Add Layers**

Use `tf.layers.dense()` to add dense layers to the model. Each dense layer is a fully connected layer, which means every input is connected to every output.

```python
model.add(tf.layers.dense({units: 1, inputShape: [1]}))
```

### **Step 3: Compile the Model**

The compile step defines how the model will be trained. You set the optimizer (like "SGD") and the loss function (like "meanSquaredError").

```python
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})
```

### **Step 4: Train the Model**

Finally, you fit the model using training data. TensorFlow will automatically adjust the model's weights and biases over multiple epochs (training iterations).

```python
xs = tf.tensor([1, 2, 3, 4])
ys = tf.tensor([2, 4, 6, 8])  # Target output
model.fit(xs, ys, {epochs: 50})  # Train for 50 epochs
```

This process is at the heart of all neural network models you will create in TensorFlow.

---

## 5. Projects to Build

Learning theory is great, but real learning happens when you apply it. Today, you identified four essential projects to work on:

- **PoseNet**: A pre-trained model for pose detection (like tracking human body positions).
- **MobileNet**: A lightweight image classification model that works well on mobile devices.
- **FaceMesh**: A model that identifies facial landmarks (like eyes, nose, and mouth) in an image.
- **Canvas Trial**: An experiment to use the Canvas API for animations and visualizations while integrating machine learning models.

Working on these projects will enhance your skills in both practical machine learning and interactive front-end development.

---

## 6. Vocabulary You Learned

| **Word**           | **Meaning / Usage**                                                 |
| ------------------ | ------------------------------------------------------------------- |
| **Prerequisites**  | Essential knowledge, skills, or resources required before starting. |
| **Leverage**       | To make use of a resource to your advantage.                        |
| **Prior**          | Describes something that happened earlier or previously.            |
| **Let's go ahead** | A common phrase used to continue or begin a task.                   |
| **Prior to**       | A phrase meaning "before." Example: "Prior to the exam."            |
| **Prior meeting**  | A meeting that occurred earlier or before another event.            |

These vocabulary words are common in technical discussions, meetings, and programming documentation.

---

## 7. What's Next?

Based on today's lessons, you should now have a solid understanding of:

- How to use Python libraries for machine learning (TensorFlow, PyTorch, NumPy, Scikit-learn).
- The importance of tensors, models, and neural network training.
- Practical machine learning models like classification, regression, and clustering.
- The concept of dimensionality reduction and why it is essential.
- How to start real-world projects using models like PoseNet, MobileNet, and FaceMesh.

You also expanded your vocabulary and picked up useful terms for everyday use in programming and machine learning discussions.

### **Next Steps**

- **Building small projects**: Use PoseNet, MobileNet, or FaceMesh on your local machine.
- **Practice coding daily**: Write and train small models from scratch to understand TensorFlow better.
- **Revisit core concepts**: Ensure you understand tensors, models, and operations like `tf.add()` and `tf.slice()`.

With this knowledge, you’re well on your way to becoming a skilled machine learning practitioner and deep learning expert. Keep pushing forward and challenge yourself with more complex models and real-world data.
