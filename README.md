# AI_agent_for_commercial_website

# AI Product Chatbot & Image Search

A web-based AI chatbot that provides **product recommendations** via **text queries** or **image uploads**. It supports **grocery items** and **e-commerce products** (jeans, sofas, T-shirts, TVs) with dynamic carousels and real-time recommendations.

---

## 🌟 Features

- **Text-based product recommendation**  
  Ask the chatbot about products like “I want apples” or “Show me sofas”.

- **Image-based product search**  
  Upload a product image and get visually similar matches.

- **Hybrid chatbot engine**  
  - Intent recognition for greetings, thanks, capabilities, and fallback  
  - Keyword & category-based product search  
  - Embedding similarity fallback for unknown queries

- **Dynamic product carousels**  
  Visualize top products for static categories and grocery items.

Make sure to install all packages including:
 fastapi 
 uvicorn 
 pydantic 
 python-multipart 
 torch 
 torchvision 
 pillow 
 transformers 
 sentence-transformers 
 numpy 

as shown in requirements.txt

use "uvicorn main:app --reload" to run the app