import random
import re
import csv
from PIL import Image
import io
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as T
from torchvision import models
from pathlib import Path
import traceback

# -----------------------
# Category synonyms
# -----------------------
CATEGORY_SYNONYMS = {
    "t-shirts": ["t-shirts", "tee", "top"],
    "jeans": ["jeans", "denim", "pants", "trousers", "slacks"],
    "sofa": ["sofa", "couch", "settee", "furniture", "sectional", "chair", "table", "desk", "cabinet", "dresser", "bench"],
    "tv": ["tv", "television", "phone", "smartphone", "mobile", "cellphone", "laptop", "notebook", "macbook", "ultrabook", "computer", "pc"],
    "fruit": [
        "apple", "golden delicious", "granny smith", "pink lady", "red delicious", "royal gala", "avocado", "banana", 
        "kiwi", "lemon", "lime", "mango", "melon", "cantaloupe", "galia melon", "honeydew melon", "watermelon", 
        "nectarine", "orange", "papaya", "passion fruit", "peach", "pear", "anjou", "conference", "kaiser", "pineapple", 
        "plum", "pomegranate", "red grapefruit", "satsumas"
    ],
    "juice": ["apple juice", "orange juice", "grapefruit juice", "tropicana", "god morgon", "bravo juice"],
    "vegetables":[
        "asparagus", "aubergine", "eggplant", "cabbage", "carrot", "carrots", "cucumber", "garlic", 
        "ginger", "leek", "mushroom", "onion", "pepper", "bell pepper", "green bell pepper", 
        "red bell pepper", "yellow bell pepper", "potato", "floury potato", "solid potato", "sweet potato",
        "red beet", "tomato", "beef tomato", "vine tomato", "regular tomato", "zucchini"
    ],
}

CATEGORY_PRIORITY = ["juice", "milk", "yoghurt", "fruit", "vegetables", "t-shirts", "jeans", "sofa", "tv"]

class HybridChatbot:
    def __init__(self, csv_path='static/GroceryStoreDataset/dataset/classes.csv', max_history=500):
        self.max_history = max_history
        self.last_recommendation = []

        # Normalize categories and synonyms
        self.category_synonyms = {}
        for cat, syns in CATEGORY_SYNONYMS.items():
            norm_cat = self.normalize(cat)
            self.category_synonyms[norm_cat] = norm_cat
            for syn in syns:
                self.category_synonyms[self.normalize(syn)] = norm_cat

        # Grocery products from CSV
        self.GROCERY_PRODUCTS = []
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    product = {
                        "class_name": self.singularize(row.get("Class Name (str)", "").strip()),
                        "coarse": self.singularize(row.get("Coarse Class Name (str)", "").strip()),
                        "img": "static/GroceryStoreDataset/dataset/" + row.get("Iconic Image Path (str)", "").strip()
                    }
                    self.GROCERY_PRODUCTS.append(product)
            print(f"[INFO] Loaded {len(self.GROCERY_PRODUCTS)} grocery products.")
        except Exception as e:
            print("[ERROR] Could not load grocery CSV:", e)

        # Static categories
        self.STATIC_CATEGORIES = {
            "jeans": [{"class_name": f"Jeans {i}", "coarse": "Jeans",
                       "img": f"static/ecommerce_products/jeans/{i}.jpg"} for i in range(1, 4)],
            "sofa": [{"class_name": f"Sofa {i}", "coarse": "Sofa",
                      "img": f"static/ecommerce_products/sofa/{i}.jpg"} for i in range(1, 4)],
            "t-shirts": [{"class_name": f"T-shirt {i}", "coarse": "T-shirt",
                          "img": f"static/ecommerce_products/tshirt/{i}.jpg"} for i in range(1, 4)],
            "tv": [{"class_name": f"TV {i}", "coarse": "TV",
                    "img": f"static/ecommerce_products/tv/{i}.jpg"} for i in range(1, 4)],
        }
        # Normalize keys
        self.STATIC_CATEGORIES = {self.normalize(k): v for k, v in self.STATIC_CATEGORIES.items()}

        # Build keywords
        self.keywords = set()
        for p in self.GROCERY_PRODUCTS:
            for word in self.tokenize(p['class_name'] + " " + p['coarse']):
                self.keywords.add(word.lower())
        for cat in self.STATIC_CATEGORIES:
            self.keywords.add(cat.lower())

        # Load embeddings model for text search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.category_embeddings = {}
        for cat, syns in CATEGORY_SYNONYMS.items():
            norm_cat = self.normalize(cat)
            norm_syns = [self.normalize(s) for s in syns]
            self.category_embeddings[norm_cat] = self.embedding_model.encode(
                norm_syns, convert_to_tensor=True, device=self.embedding_model.device
            )

        # Combine all products for image search
        self.ALL_PRODUCTS = self.GROCERY_PRODUCTS.copy()
        for cat_products in self.STATIC_CATEGORIES.values():
            self.ALL_PRODUCTS.extend(cat_products)

        # Precompute embeddings for image search (optional: for efficiency)
        self.image_embeddings = {}  # key: product index, value: embedding tensor
        # self.image_model = torch.hub.load('pytorch/vision:v0.16.0', 'resnet18', pretrained=True)
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = torch.nn.Identity()  # remove classification head
        self.image_model.eval()
        self.image_model = self.image_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Image transform
        self.image_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Intents
        self.intents = {
            "greeting": {"patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
                         "responses": ["Hello! How can I help you today?",
                                       "Hi there! What can I do for you?", "Hey! How’s it going?"]},
            "goodbye": {"patterns": ["bye", "goodbye", "see you"],
                        "responses": ["Goodbye! Have a nice day!", "See you later!", "Bye! Take care."]},
            "thanks": {"patterns": ["thanks", "thank you", "appreciate it"],
                       "responses": ["You're welcome!", "No problem!", "Happy to help!"]},
            "capabilities": {"patterns": ["what can you do", "how can you help", "what are your features",
                                          "your capabilities", "what do you offer"],
                             "responses": [
                                 "I can help you with text-based product recommendations 📚, image-based product search 🖼️, and casual conversation 💬.",
                                 "My main features are: product recommendations, image search, and chatting with you.",
                                 "I can assist with shopping suggestions, find products by image, and have general conversations!"
                             ]},
            "fallback": {"patterns": [], "responses": ["I'm not sure I understand. Can you rephrase?",
                                                       "Sorry, I don’t know about that.",
                                                       "Hmm, interesting. Tell me more!"]}
        }

        # Intent vocab & BoW matrix
        self.vocab = set()
        for intent in self.intents.values():
            for pattern in intent["patterns"]:
                for word in self.tokenize(pattern):
                    self.vocab.add(word.lower())
        self.vocab = list(self.vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}

        self.intent_matrix = {}
        for intent_name, intent_data in self.intents.items():
            matrix = [self.text_to_bow(pattern) for pattern in intent_data["patterns"]]
            self.intent_matrix[intent_name] = np.array(matrix)

        # Small generative model for fallback
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_safetensors=True).to(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.chat_history_ids = None

    # -----------------------
    # Helper methods
    # -----------------------
    def singularize(self, word):
        if word.endswith('s') and len(word) > 3:
            return word[:-1]
        # elif word.endswith('es') and not word.endswith('ses'):
        #     return word[:-2]
        # elif word.endswith('ies'):
        #     return word[:-3] + 'y'
        else:
            return word

    def tokenize(self, text):
        tokens = re.findall(r'\w+(?:-\w+)?', text.lower())
        return [self.singularize(re.sub(r'[\s\-_]', '', t)) for t in tokens]

    def normalize(self, text):
        return re.sub(r'[\s\-_]', '', text.lower())

    def text_to_bow(self, text):
        vec = np.zeros(len(self.vocab))
        for word in self.tokenize(text):
            if word in self.word2idx:
                vec[self.word2idx[word]] += 1
        return vec

    def predict_intent(self, user_input):
        vec = self.text_to_bow(user_input)
        best_intent, best_score = "fallback", 0
        for intent_name, matrix in self.intent_matrix.items():
            if len(matrix) == 0:
                continue
            scores = matrix.dot(vec) / (np.linalg.norm(matrix, axis=1) * (np.linalg.norm(vec) + 1e-6))
            score = float(np.max(scores)) if len(scores) > 0 else 0
            if score > best_score:
                best_intent, best_score = intent_name, score
        return best_intent if best_score > 0.25 else "fallback"

    def generate_fallback(self, user_input):
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt").to(self.model.device)
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids
        with torch.no_grad():
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=300,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9
            )
        # Truncate history
        if self.chat_history_ids.shape[-1] > self.max_history:
            self.chat_history_ids = self.chat_history_ids[:, -self.max_history:]
        reply = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return reply

    # -----------------------
    # Recommendation
    # -----------------------
    def recommend_products(self, user_input=None, top_n=5):
        self.last_recommendation = []
        matched_products = []

        if not user_input:
            return {"text": "Please tell me what kind of product you're looking for!", "products": []}

        # 1. tokenize and normalize user input
        norm_input_tokens = [self.normalize(t) for t in self.tokenize(user_input)]

        # 2. prioritize categories
        CATEGORY_PRIORITY = ["juice", "milk", "yoghurt", "fruit", "vegetables"]  # higher to lower

        # 3. match grocery products
        for cat in CATEGORY_PRIORITY:
            for product in self.GROCERY_PRODUCTS:
                product_tokens = [self.normalize(w) for w in self.tokenize(product['class_name'] + " " + product['coarse'])]
                # if any normalized user token is in normalized product tokens
                if any(u in product_tokens for u in norm_input_tokens):
                    # check category matches, singularize + normalize coarse class name
                    norm_coarse = self.normalize(self.singularize(product['coarse']))
                    product_category = self.category_synonyms.get(norm_coarse, None)
                    if product_category == cat:
                        matched_products.append(product)
            if matched_products:
                matched_products = matched_products[:top_n]
                break  # stop after first priority category match

        # 4. check static category synonyms if no grocery match
        if not matched_products:
            for token in norm_input_tokens:
                if token in self.category_synonyms:
                    category = self.category_synonyms[token]
                    matched_products = self.STATIC_CATEGORIES.get(category, [])[:top_n]
                    break

        # 5. fallback to embedding similarity if still empty
        if not matched_products:
            user_emb = self.embedding_model.encode(user_input, convert_to_tensor=True, device=self.embedding_model.device)
            best_cat = None
            best_score = 0.7
            for cat, emb in self.category_embeddings.items():
                score = util.pytorch_cos_sim(user_emb, emb).max().item()
                if score > best_score:
                    best_score = score
                    best_cat = cat
            if best_cat:
                matched_products = self.STATIC_CATEGORIES.get(best_cat, [])[:top_n]

        # 6. random fallback
        if not matched_products:
            matched_products = random.sample(self.GROCERY_PRODUCTS, min(top_n, len(self.GROCERY_PRODUCTS)))

        selected = random.sample(matched_products, min(top_n, len(matched_products)))
        self.last_recommendation = selected
        messages = [f"{p['class_name']} ({p['coarse']}) - see here: {p['img']}" for p in selected]
        return {"text": "I recommend these products: " + " | ".join(messages), "products": selected}

    # -----------------------
    # Get response for text search
    # -----------------------
    def get_response(self, user_input):
        norm_input_tokens = [self.normalize(w) for w in self.tokenize(user_input)]
        norm_input_text = self.normalize(user_input)

        # 1. Check static and grocery categories first
        for token in norm_input_tokens + [norm_input_text]:
            if token in self.category_synonyms:
                return self.recommend_products(user_input)

        # 2. Check keywords match
        if any(token in self.keywords for token in norm_input_tokens):
            return self.recommend_products(user_input)

        # 3. Predict intent
        intent = self.predict_intent(user_input)
        if intent != "fallback":
            return {"text": random.choice(self.intents[intent]["responses"])}

        # 4. Fallback: try recommending products anyway
        return self.recommend_products(user_input)

    # -----------------------
    # Get response for image search
    # -----------------------
    def get_response_from_image(self, image_bytes, top_n=5):
        try:
            device = next(self.image_model.parameters()).device
            # Open uploaded image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = self.image_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                query_emb = self.image_model(img_tensor)

            scores = []
            base_dir = Path(__file__).resolve().parent

            for idx, product in enumerate(self.ALL_PRODUCTS):
                try:
                    prod_img_path = base_dir / product["img"]
                    if not prod_img_path.exists():
                        print(f"[WARN] Missing image: {prod_img_path}")
                        continue

                    # Compute embedding only if not already cached
                    if idx not in self.image_embeddings:
                        try:
                            prod_img = Image.open(prod_img_path).convert('RGB')
                            prod_tensor = self.image_transform(prod_img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                emb = self.image_model(prod_tensor)
                            self.image_embeddings[idx] = emb
                        except Exception as e:
                            print(f"[WARN] Failed to load image {prod_img_path}: {e}")
                            continue

                    # Cosine similarity
                    sim = torch.nn.functional.cosine_similarity(query_emb, self.image_embeddings[idx], dim=1)
                    scores.append((sim.item(), product))

                except Exception as e:
                    print(f"[WARN] Failed processing product {product.get('img', '?')}: {e}")
                    continue

            # Sort top N
            scores.sort(key=lambda x: x[0], reverse=True)
            top_products = [p for _, p in scores[:top_n]]

            if top_products:
                messages = [f"{p['class_name']} ({p['coarse']}) - see: {p['img']}" for p in top_products]
                return {"text": "I found these products matching your image: " + " | ".join(messages),
                        "products": top_products}
            else:
                return {"text": "Sorry, I couldn’t find visually similar products.", "products": []}

        except Exception as e:
            print("[ERROR] Image search failed:", e)
            traceback.print_exc()  # Add this line
            return {"text": "Sorry, I couldn't process your image.", "products": []}