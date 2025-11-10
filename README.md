# AI Agent for Commercial Website

## AI Product Chatbot & Image Search

A web-based multimodal **AI chatbot** that **combines semantic text search, image similarity retrieval, and intent-driven dialogue to recommend products intelligently** via both **text queries** and **image uploads**.  
It supports a variety of products such as **grocery items**, **jeans**, **sofas**, **T-shirts**, and **TVs**, with **dynamic carousels**. 

---

## Features

- **Text-based product recommendation**  
  Ask the chatbot about products like “I want apples” or “Show me sofas”.

- **Image-based product search**  
  Upload a product image and get visually similar matches.

- **Hybrid chatbot engine**  
  Intent recognition for greetings, thanks, capabilities, and fallback  
  Combines intent recognition, keyword matching, and embedding similarity for flexible responses.


- **Product carousels**  
  Visualize products for static categories and grocery items.

The system combines **Natural Language Processing (NLP)**, **Computer Vision**, and **FastAPI-based web services** to provide intelligent product recommendations.

---

## Technology Stack and Design Decisions

| Component               | Technology  | Reason for Choice |
|-------------------------|-------------|-------------------|
| **Backend Framework**   | **FastAPI** | Lightweight, asynchronous, and perfect for high-performance AI APIs |
| **Server**              | **Uvicorn** | ASGI-compatible server optimized for FastAPI apps |
| **Model Management**    | **PyTorch**, **Torchvision** | Industry-standard deep learning framework for computer vision |
| **NLP Models**          | **Transformers**, **Sentence-Transformers** | Powerful pretrained models for semantic similarity and intent understanding |
| **Data Validation**     | **Pydantic** | Simple and reliable schema enforcement for API requests/responses |
| **File Handling**       | **python-multipart** | Enables image upload via API endpoints |
| **Image Processing**    | **Pillow (PIL)** | Efficient image manipulation and format handling |
| **Computation**         | **NumPy** | Fast numerical operations for embeddings and similarity metrics |
| **Frontend**            | **HTML + JavaScript** | Simple, responsive interface for user interaction |

---

## System Architecture


1. **Frontend Interface**  
   - Users can send text queries or upload images.  
   - Results are displayed dynamically as product carousels.  

2. **FastAPI Backend (`main.py`)**  
   - Handles both `/recommend` (text query) and `/image-search` (image upload) endpoints.  
   - Returns structured JSON responses to the frontend.  

3. **AI Agent (`chat_agent.py`)**  
  
   Implements a Hybrid AI pipeline combining intent classification, NLP, and image-based retrieval.

   - Text Processing: 
      Uses SentenceTransformer (all-MiniLM-L6-v2) for semantic similarity and category matching. 
      Employs DialoGPT-small for natural fallback and conversational responses.
      Performs rule-based keyword matching, intent detection, and synonym normalization across product categories.
   - Image Analysis:
      Uses a pretrained ResNet-18 CNN backbone (from torchvision.models) for image embeddings.
      Compares uploaded images against a preloaded dataset to find visually similar items.
   - Product Knowledge Base:
      Loads grocery and e-commerce product data from a CSV file and static image directories.
      Dynamically computes and caches image embeddings for efficiency.
   - Conversational fallback generation:
      Uses DialoGPT for natural user interaction.

4. **Response Handler**  
   - Classifies output type (`string_response`, `dict_response`, or `exception`)  
   - Enables seamless communication between the AI logic and the frontend.

---

## API Endpoint

| Endpoint        | Method | Input                       | Output                                                   | Description                                               |
| --------------- | ------ | --------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| `/`             | GET    | None                        | HTML page                                                | Serves the frontend `index.html` if available             |
| `/ping`         | GET    | None                        | {"status": "ok"}                                         | Health check endpoint                                     |
| `/recommend`    | POST   | {"text": "I want apples"}   | {"text": "...", "products": [...]}                       | Returns product recommendations based on a text query     |
| `/image-search` | POST   | Image file (via form-data)  | {"text": "...", "products": [...], "condition": "..."}   | Returns visually similar products from the uploaded image |

---

Make sure to install all packages including:
 fastapi, uvicorn, pydantic, python-multipart, torch, torchvision, pillow, transformers, sentence-transformers, numpy 

as shown in requirements.txt

use "uvicorn main:app --reload" to run the app,
then open your browser and go to: http://127.0.0.1:8000

---

## Datasets

This project uses two datasets for product recommendation and image search. You can download them manually from Kaggle or programmatically using `kagglehub`.

### 1. Grocery Store Image Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/amoghmisra27/grocery)  
- **Description:** Contains images of various grocery items (fruits, vegetables, juices, etc.) used for text- and image-based recommendations.

### 2. E-commerce Products Image Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/sunnykusawa/ecommerce-products-image-dataset?resource=download)  
- **Description:** Contains images of e-commerce products like jeans, sofas, T-shirts, and TVs for static category recommendations.

### Download Datasets with `kagglehub`

```python
import kagglehub

# Download latest Grocery Store dataset
grocery_path = kagglehub.dataset_download("amoghmisra27/grocery")
print("Path to grocery dataset files:", grocery_path)

# Download latest E-commerce Products dataset
ecommerce_path = kagglehub.dataset_download("sunnykusawa/ecommerce-products-image-dataset")
print("Path to e-commerce dataset files:", ecommerce_path)
```
