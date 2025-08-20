# IDE Setup Instructions

## VS Code Setup

### 1. Python Interpreter Selection
1. Open VS Code in the backend directory
2. Press `Ctrl+Shift+P`
3. Type "Python: Select Interpreter"
4. Choose: `./venv/bin/python`

### 2. Reload Window
1. Press `Ctrl+Shift+P`
2. Type "Developer: Reload Window"
3. Press Enter

### 3. Verify Setup
After reload, you should see:
- No import errors for:
  - `redis.asyncio`
  - `sqlalchemy`
  - `sqlalchemy.ext.asyncio`
  - `sqlalchemy.orm`
  - `numpy`
  - `boto3`
  - `botocore.exceptions`
  - `PyPDF2`
  - `scipy`
  - `app.core.cache`

## PyCharm Setup

### 1. Python Interpreter Configuration
1. Go to `File` → `Settings` → `Project` → `Python Interpreter`
2. Click `Add Interpreter` → `Existing Environment`
3. Set path to: `/home/ela/Desktop/yoladgunew/backend/venv/bin/python`

### 2. Mark Directories as Sources
1. Right-click on `app` folder
2. Select `Mark Directory as` → `Sources Root`

### 3. Invalidate Caches
1. Go to `File` → `Invalidate Caches and Restart`
2. Click `Invalidate and Restart`

## Troubleshooting

If you still see import errors:

1. **Check Python Path:**
   ```bash
   cd /home/ela/Desktop/yoladgunew/backend
   source venv/bin/activate
   python -c "import sys; print(sys.path)"
   ```

2. **Verify Virtual Environment:**
   ```bash
   which python
   # Should show: /home/ela/Desktop/yoladgunew/backend/venv/bin/python
   ```

3. **Test Imports:**
   ```bash
   python -c "import redis.asyncio; import sqlalchemy; print('OK')"
   ```

4. **Reinstall Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Files Created/Modified

- ✅ `.vscode/settings.json` - VS Code configuration
- ✅ `pyrightconfig.json` - Pyright/Pylance configuration
- ✅ `app/core/cache.py` - Cache module
- ✅ `setup.py` - Python path setup
- ✅ `requirements.txt` - Updated with type stubs
- ✅ `IDE_SETUP.md` - This file

## Type Stubs Installed

- `types-sqlalchemy` - SQLAlchemy type hints
- `types-redis` - Redis type hints
- `types-boto3` - Boto3 type hints
