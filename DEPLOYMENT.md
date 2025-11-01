# VaaniSetu Deployment Guide

## Architecture Overview
- **Frontend**: React + Vite → Supabase Storage (static hosting)
- **Backend**: FastAPI Python → Railway/Render/Fly.io (or similar)

## Why This Split?
Supabase doesn't support Python backends directly. We'll host:
- Static frontend on Supabase Storage (free, fast CDN)
- Backend API on a Python-compatible platform

---

## Option 1: Deploy to Supabase + Railway (Recommended)

### Step 1: Deploy Backend to Railway

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize Railway project** (in project root):
   ```bash
   railway init
   ```

4. **Deploy backend**:
   ```bash
   cd backend
   railway up
   ```

5. **Note your backend URL** (e.g., `https://your-app.railway.app`)

### Step 2: Configure Frontend for Production

1. **Create `.env.production` in `frontend/`**:
   ```env
   VITE_API_URL=https://your-app.railway.app
   ```

2. **Update WebSocket URL in your frontend code** to use Railway URL

### Step 3: Build Frontend

```bash
cd frontend
npm install
npm run build
```

This creates a `frontend/dist` folder with static files.

### Step 4: Deploy Frontend to Supabase

1. **Create Supabase project** at https://supabase.com

2. **Create Storage Bucket**:
   - Go to Storage in Supabase Dashboard
   - Create new bucket named `website`
   - Make it **public**

3. **Upload frontend files**:
   - Upload entire `frontend/dist` folder to the bucket
   - Set `index.html` as the default file

4. **Get your website URL**:
   ```
   https://[project-ref].supabase.co/storage/v1/object/public/website/index.html
   ```

5. **Optional: Use custom domain** via Supabase settings

### Step 5: Update CORS in Backend

Update `backend/app/main.py` CORS origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://[your-project-ref].supabase.co",  # Add this
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Redeploy backend: `railway up`

---

## Option 2: Deploy to Supabase + Render

### Step 1: Deploy Backend to Render

1. **Create `render.yaml`** in project root (already created in this repo)

2. **Push to GitHub** (Render requires Git)

3. **Connect Render**:
   - Go to https://render.com
   - New → Web Service
   - Connect your GitHub repo
   - Render auto-detects `render.yaml`

4. **Note your backend URL** (e.g., `https://vaanisetu.onrender.com`)

### Step 2-5: Same as Railway
Follow Railway steps 2-5, but use Render URL instead.

---

## Option 3: All-in-One Alternative (Vercel/Netlify)

If you prefer simpler deployment:

1. **Deploy to Vercel** (supports both frontend + serverless Python):
   ```bash
   npm install -g vercel
   vercel
   ```

2. **Or deploy to Netlify** with functions

---

## Environment Variables for Backend

Set these on Railway/Render:

```env
PORT=8000
PYTHONUNBUFFERED=1
# Add any API keys for translation services
OLLAMA_HOST=http://localhost:11434  # If using Ollama
```

---

## Monitoring & Logs

- **Railway**: `railway logs`
- **Render**: View logs in dashboard
- **Frontend**: Supabase Storage has access logs

---

## Costs

- **Supabase**: Free tier includes 1GB storage
- **Railway**: $5/month after 500 hours free trial
- **Render**: Free tier (spins down after 15 min inactivity)

---

## Troubleshooting

### CORS Errors
- Ensure backend CORS includes your Supabase domain
- Check browser console for exact origin being blocked

### WebSocket Connection Fails
- WebSocket URLs should use `wss://` (not `ws://`) in production
- Update frontend to use: `wss://your-backend.railway.app/ws`

### 404 on Refresh
- Supabase Storage doesn't support SPA routing
- Consider using hash routing or deploying frontend elsewhere (Vercel/Netlify handle this better)

---

## Next Steps

1. Set up CI/CD with GitHub Actions
2. Add monitoring (Sentry, LogRocket)
3. Optimize bundle size
4. Add caching headers
5. Set up custom domain
