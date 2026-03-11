# 📤 GitHub Upload Guide

## 🎯 Quick Upload (Automated)

### Option 1: Using the Upload Script ⭐ Recommended

```bash
# Make script executable
chmod +x upload_to_github.sh

# Run the upload script
./upload_to_github.sh
```

This will:
1. ✅ Initialize git repository (if needed)
2. ✅ Add all files
3. ✅ Commit changes
4. ✅ Set up remote origin
5. ✅ Push to GitHub main branch

---

### Option 2: Manual Upload

#### Step 1: Initialize Git (if not done)
```bash
git init
```

#### Step 2: Add All Files
```bash
git add .
```

#### Step 3: Commit
```bash
git commit -m "Initial commit: Complete Customer Churn Prediction Dashboard"
```

#### Step 4: Add Remote Repository
```bash
git remote add origin https://github.com/Amritesh0007/Customer_Churn_Prediction.git
```

#### Step 5: Rename Branch to Main
```bash
git branch -M main
```

#### Step 6: Push to GitHub
```bash
git push -u origin main
```

---

## 🔐 Authentication

GitHub may require authentication. You have two options:

### Option A: HTTPS with Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token when prompted for password:
   ```bash
   Username: your_username
   Password: your_personal_access_token
   ```

### Option B: SSH Keys (Recommended)

1. **Generate SSH Key** (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Add SSH Key to GitHub**:
   - Copy your public key: `cat ~/.ssh/id_ed25519.pub`
   - Go to GitHub Settings → SSH and GPG keys
   - Click "New SSH key" and paste

3. **Change Remote to SSH**:
   ```bash
   git remote set-url origin git@github.com:Amritesh0007/Customer_Churn_Prediction.git
   ```

4. **Push**:
   ```bash
   git push -u origin main
   ```

---

## 📦 What Gets Uploaded

### ✅ Included Files:
- `web_dashboard.py` - Main Streamlit app
- `train_model.py` - Model training script
- `dataset.py` - Data generator
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `LICENSE` - MIT License
- `.gitignore` - Git ignore rules
- `sample_customer_data.csv` - CSV template
- All utility scripts
- Model artifacts (*.joblib files)
- Documentation files

### ❌ Excluded Files (by .gitignore):
- `__pycache__/` folders
- `.pyc` files
- `.DS_Store` (macOS)
- Virtual environments
- IDE files (.vscode, .idea)
- Jupyter checkpoints
- Large data files (except sample)

---

## 🎨 Post-Upload Steps

### 1. Verify Repository
Visit: https://github.com/Amritesh0007/Customer_Churn_Prediction

Check that all files are present:
- ✅ README.md displays correctly
- ✅ All Python scripts uploaded
- ✅ Model files included
- ✅ Sample CSV present

### 2. Protect Your Main Branch (Optional but Recommended)
1. Go to repository Settings → Branches
2. Click "Add branch protection rule"
3. Branch name pattern: `main`
4. Check "Require pull request reviews before merging"

### 3. Add Repository Topics
1. Go to repository main page
2. Click the gear icon next to "About"
3. Add topics:
   - `churn-prediction`
   - `machine-learning`
   - `xgboost`
   - `streamlit`
   - `dashboard`
   - `customer-analytics`
   - `python`

### 4. Enable GitHub Pages (Optional)
If you want to showcase screenshots:
1. Go to Settings → Pages
2. Source: Deploy from branch
3. Branch: main → `/docs` folder (create if needed)
4. Save

---

## 🔄 Updating the Repository

### Making Changes Locally
```bash
# Make your changes to files
git add <changed_files>
git commit -m "Description of changes"
git push origin main
```

### Example: Update Model
```bash
python train_model.py  # Retrain with new data
git add churn_model.joblib
git commit -m "Update model with Q1 2026 data"
git push origin main
```

---

## 📊 Repository Size Optimization

If the repository becomes too large (>100MB):

### Option 1: Store Models Separately
```bash
# Remove large model files from git
git rm --cached *.joblib
git commit -m "Remove large model files"

# Create models folder in .gitignore
echo "*.joblib" >> .gitignore
```

Then host models on:
- Google Drive
- AWS S3
- Hugging Face
- GitHub Releases

### Option 2: Use Git LFS (Large File Storage)
```bash
# Install Git LFS
brew install git-lfs  # macOS
sudo apt-get install git-lfs  # Linux

# Setup LFS
git lfs install

# Track large files
git lfs track "*.joblib"
git lfs track "*.csv"

# Add and commit
git add .gitattributes
git add *.joblib
git commit -m "Add models with LFS"
git push origin main
```

---

## 🎉 Success Checklist

After uploading, verify:

- [ ] Repository is accessible at: https://github.com/Amritesh0007/Customer_Churn_Prediction
- [ ] README.md displays correctly with formatting
- [ ] All Python files are present
- [ ] requirements.txt is complete
- [ ] LICENSE file exists
- [ ] .gitignore is configured
- [ ] Model files are included (<100MB total)
- [ ] Sample CSV template is present
- [ ] Documentation files uploaded

---

## 🐛 Troubleshooting

### Error: "remote: Repository not found"
**Solution**: Make sure the repository exists on GitHub first. Create it empty, then push.

### Error: "fatal: remote origin already exists"
**Solution**: 
```bash
git remote remove origin
git remote add origin https://github.com/Amritesh0007/Customer_Churn_Prediction.git
```

### Error: "Updates were rejected because the remote contains work that you do not have"
**Solution**: 
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Error: "File size exceeds limit"
**Solution**: Remove large files or use Git LFS (see above)

---

## 📞 Need Help?

### GitHub Resources:
- [Creating a new repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [Adding an existing project to GitHub](https://docs.github.com/en/get-started/importing-your-projects-to-github/importing-source-code-to-github/adding-an-existing-project-to-github-using-the-command-line)
- [Authentication to GitHub](https://docs.github.com/en/authentication)

### Contact:
- Open an issue on the repository
- Email: amriteshkumar475@gmail.com

---

## 🚀 Ready to Upload?

```bash
# Quick start
./upload_to_github.sh

# Or manually
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Amritesh0007/Customer_Churn_Prediction.git
git branch -M main
git push -u origin main
```

**Your professional churn prediction dashboard will be live on GitHub!** 🎊
