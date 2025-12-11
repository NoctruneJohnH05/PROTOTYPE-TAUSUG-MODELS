# 🚀 Complete Deployment Guide: Tausug Autocomplete to Render

This guide will walk you through deploying your Django application with machine learning models to **Render.com** using their **free tier**.

---

## 📋 **Prerequisites**

Before starting, make sure you have:
- ✅ GitHub account (your repository: `NoctruneJohnH05/PROTOTYPE-TAUSUG-MODELS`)
- ✅ Git installed on your computer
- ✅ All your model files in the project (`.keras`, `.pt`, `.model` files)

---

## 🎯 **Overview: What We'll Do**

1. **Install Git LFS** - To handle large model files (>100MB)
2. **Configure Git LFS** - Tell Git which files are "large"
3. **Create deployment configuration files** - Tell Render how to run your app
4. **Update Django settings** - Make it production-ready
5. **Push to GitHub** - Upload everything including models
6. **Deploy on Render** - Connect GitHub and go live!

---

## 📦 **STEP 1: Install Git LFS (Large File Storage)**

### **What is Git LFS?**
Git LFS is a tool that handles large files (like your ML models) efficiently. Without it, GitHub will reject files over 100MB.

### **How to Install:**

**On Windows:**
1. Download Git LFS installer: https://git-lfs.github.com/
2. Run the installer (it's just a simple `.exe` file)
3. Open PowerShell in your project folder
4. Run this command:
   ```powershell
   git lfs install
   ```
   You should see: `Git LFS initialized.`

**To verify it worked:**
```powershell
git lfs version
```
You should see something like: `git-lfs/3.4.0`

---

## 🔧 **STEP 2: Configure Git LFS to Track Model Files**

### **What this does:**
Tells Git LFS to handle your large model files (`.keras`, `.pt`, `.model`) separately from regular code files.

### **Steps:**

1. **Open PowerShell** in your project root folder:
   ```powershell
   cd c:\Users\Tong\Documents\GitHub\PROTOTYPE-TAUSUG-MODELS
   ```

2. **Track model file types:**
   ```powershell
   git lfs track "*.keras"
   git lfs track "*.pt"
   git lfs track "*.model"
   git lfs track "*.h5"
   git lfs track "*.bin"
   ```

3. **This creates a file called `.gitattributes`** - Add it to Git:
   ```powershell
   git add .gitattributes
   git commit -m "Configure Git LFS for model files"
   ```

### **What happens:**
- A file called `.gitattributes` is created in your project
- It tells Git: "These file types are large, handle them with LFS"

---

## 📝 **STEP 3: Create Deployment Configuration Files**

You need to create **3 new files** in your project root. Here's what each one does and how to create them:

---

### **File 1: `requirements.txt`**

**What it does:** Lists all Python packages Render needs to install.

**Where to create it:** Project root (`c:\Users\Tong\Documents\GitHub\PROTOTYPE-TAUSUG-MODELS\requirements.txt`)

**Content:**
```txt
Django>=4.2,<5.0
gunicorn>=21.2.0
sentencepiece>=0.1.99
numpy>=1.24.0
keras>=3.0.0
tensorflow>=2.15.0
torch>=2.1.0
```

**How to create:**
1. Open VS Code
2. Right-click in the file explorer → New File
3. Name it `requirements.txt` (in the root folder)
4. Copy-paste the content above
5. Save

---

### **File 2: `build.sh`**

**What it does:** Script that runs when Render builds your app (installs packages, collects static files, runs migrations).

**Where to create it:** Project root (`c:\Users\Tong\Documents\GitHub\PROTOTYPE-TAUSUG-MODELS\build.sh`)

**Content:**
```bash
#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Running database migrations..."
python manage.py migrate

echo "Build completed successfully!"
```

**How to create:**
1. Create new file: `build.sh`
2. Copy-paste the content above
3. Save

---

### **File 3: `render.yaml`**

**What it does:** Tells Render how to configure your web service (Python version, commands to run, environment variables).

**Where to create it:** Project root (`c:\Users\Tong\Documents\GitHub\PROTOTYPE-TAUSUG-MODELS\render.yaml`)

**Content:**
```yaml
services:
  - type: web
    name: tausug-autocomplete
    runtime: python
    plan: free
    buildCommand: "./build.sh"
    startCommand: "gunicorn mysite.wsgi:application"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: SECRET_KEY
        generateValue: true
      - key: DEBUG
        value: False
```

**How to create:**
1. Create new file: `render.yaml`
2. Copy-paste the content above
3. Save

---

## ⚙️ **STEP 4: Update Django Settings for Production**

### **What this does:**
Makes your Django app work properly in production (allows Render's domain, configures static files).

### **File to edit:** `mysite/settings.py`

### **Changes to make:**

1. **Find this line** (around line 26):
   ```python
   ALLOWED_HOSTS = []
   ```

2. **Replace it with:**
   ```python
   import os

   # Allow Render domains and localhost
   ALLOWED_HOSTS = [
       '.onrender.com',  # Render domains
       'localhost',
       '127.0.0.1',
   ]

   # Read DEBUG from environment variable (False in production)
   DEBUG = os.environ.get('DEBUG', 'True') == 'True'
   ```

3. **Find this line** (around line 119):
   ```python
   STATIC_URL = 'static/'
   ```

4. **Add these lines RIGHT AFTER it:**
   ```python
   # Static files configuration for production
   STATIC_ROOT = BASE_DIR / 'staticfiles'
   
   # Middleware for serving static files (add if not present)
   MIDDLEWARE = [
       'django.middleware.security.SecurityMiddleware',
       'whitenoise.middleware.WhiteNoiseMiddleware',  # Add this line
       # ... rest of your middleware
   ]
   ```

5. **Update `requirements.txt`** to include WhiteNoise (for serving static files):
   git add .

## 🌐 **STEP 5: Push Everything to GitHub**

### **What this does:**
Uploads all your code and models to GitHub so Render can access them.

### **Steps:**

1. **Open PowerShell** in your project folder

2. **Add all files:**
   ```powershell
   git add .
   ```

3. **Commit changes:**
   ```powershell
   git commit -m "Add deployment configuration for Render"
   ```

4. **Push to GitHub:**
   ```powershell
   git push origin john1
   ```

### **What happens:**
- Git LFS automatically uploads your large model files separately
- All code and configuration files are uploaded to GitHub
- This might take a few minutes depending on model file sizes

---

## 🎉 **STEP 6: Deploy on Render**

### **Steps:**

1. **Go to Render.com:**
   - Visit: https://render.com/
   - Click **"Get Started for Free"**
   - Sign up with your GitHub account

2. **Create New Web Service:**
   - Click **"New +"** button (top right)
   - Select **"Web Service"**

3. **Connect Your Repository:**
   - Grant Render access to your GitHub repositories
   - Select: `NoctruneJohnH05/PROTOTYPE-TAUSUG-MODELS`
   - Branch: `john1`

4. **Configure Service (Render will auto-detect most settings):**
   - **Name:** `tausug-autocomplete` (or whatever you want)
   - **Region:** Choose closest to you (e.g., Oregon/Singapore)
   - **Branch:** `john1`
   - **Runtime:** Python 3
   - **Build Command:** `./build.sh`
   - **Start Command:** `gunicorn mysite.wsgi:application`
   - **Plan:** Free

5. **Environment Variables (Important!):**
   - Render should auto-add `SECRET_KEY` from `render.yaml`
   - Make sure `DEBUG` is set to `False`

6. **Click "Create Web Service"**

### **What happens:**
- Render clones your GitHub repository
- Downloads model files via Git LFS
- Runs `build.sh` (installs packages, collects static files)
- Starts your Django app with Gunicorn
- Gives you a URL like: `https://tausug-autocomplete.onrender.com`

---

## ⏱️ **STEP 7: Wait for Deployment**

### **What to expect:**

1. **First deployment takes 10-15 minutes** because:
   - Installing TensorFlow, PyTorch, etc. (~5 min)
   - Downloading your model files (~5 min)
   - Running migrations and collecting static files (~1 min)

2. **You can watch progress** in Render's dashboard:
   - Click on your service
   - Go to "Logs" tab
   - You'll see real-time build output

3. **When it says "Your service is live 🎉":**
   - Your app is deployed!
   - Click the URL at the top to visit your site

---

## ⚠️ **Important: Free Tier Limitations**

### **What you need to know:**

1. **Cold Starts (~30-60 seconds):**
   - After 15 minutes of inactivity, Render "spins down" your app
   - Next request will take 30-60 seconds to "wake up"
   - After that, it's fast again

2. **Memory Limit (512MB):**
   - Your app has limited RAM
   - Only ONE model should load at a time (your code already does this with caching)

3. **750 Hours/Month:**
   - Free tier gives 750 hours of runtime per month
   - That's about 31 days (basically always running)

---

## 🧪 **STEP 8: Test Your Deployed App**

1. **Visit your Render URL:**
   ```
   https://your-app-name.onrender.com/autocomplete/
   ```

2. **First load will be slow** (~30-60 seconds) because:
   - Models need to load into memory
   - This happens once, then it's cached

3. **Test autocomplete:**
   - Type some Tausug words
   - Check if predictions work
   - Try all 3 models (LSTM, Bidirectional, GRU)

---

## 🐛 **Troubleshooting**

### **Problem: Build fails with "File too large"**
**Solution:** Make sure Git LFS is installed and tracking your model files:
```powershell
git lfs track "*.keras" "*.pt" "*.model"
git add .gitattributes
git commit -m "Fix LFS tracking"
git push
```

### **Problem: "Application failed to start"**
**Solution:** Check Render logs for errors:
1. Go to Render dashboard
2. Click your service
3. Click "Logs" tab
4. Look for error messages

### **Problem: 502 Bad Gateway**
**Solution:** App is probably out of memory. Check logs. You may need to:
- Use Render's paid tier ($7/month) for more RAM
- Optimize model loading (only load one model at a time)

### **Problem: Static files not loading (CSS broken)**
**Solution:** Make sure you added WhiteNoise to settings and ran:
```python
python manage.py collectstatic
```

---

## 📊 **Summary: What Each File Does**

| File | Purpose |
|------|---------|
| `.gitattributes` | Tells Git LFS which files are "large" |
| `requirements.txt` | Lists Python packages to install |
| `build.sh` | Script that runs during deployment (install, migrate, collect static) |
| `render.yaml` | Configuration file for Render (Python version, commands, environment) |
| `mysite/settings.py` | Updated for production (ALLOWED_HOSTS, STATIC_ROOT, DEBUG from env) |

---

## 🎓 **Deployment Flow Diagram**

```
1. You push code to GitHub
        ↓
2. Render detects changes
        ↓
3. Render clones repository
        ↓
4. Git LFS downloads model files
        ↓
5. Render runs build.sh:
   - Install Python packages
   - Collect static files
   - Run database migrations
        ↓
6. Render starts app with:
   gunicorn mysite.wsgi:application
        ↓
7. Your app is live! 🎉
```

---

## 🆘 **Need Help?**

If you get stuck:
1. Check Render logs (most issues show up there)
2. Verify all files are created correctly
3. Make sure Git LFS is tracking your models:
   ```powershell
   git lfs ls-files
   ```
   You should see your `.keras`, `.pt`, `.model` files listed

---

## ✅ **Checklist: Before Deploying**

- [ ] Git LFS installed (`git lfs version` works)
- [ ] Model files tracked (`.gitattributes` exists)
- [ ] `requirements.txt` created
- [ ] `build.sh` created
- [ ] `render.yaml` created
- [ ] `settings.py` updated (ALLOWED_HOSTS, STATIC_ROOT, DEBUG)
- [ ] All changes committed and pushed to GitHub
- [ ] Render account created
- [ ] Repository connected to Render

---

## 🚀 **Ready to Deploy?**

Follow the steps above in order, and you'll have your Tausug autocomplete app live on the internet in about 20-30 minutes!

Good luck! 🎉
