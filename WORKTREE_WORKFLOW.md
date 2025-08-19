# Proper Git Worktree Workflow

## Current Setup
- **Main worktree**: `/Users/diyagamah/Documents/ufc-predictor` (main branch)
- **Feature worktree**: `/Users/diyagamah/Documents/ufc-predictor-feature` (new-feature branch)

## ✅ CORRECT WORKFLOW

### 1. Work in your feature worktree
```bash
cd /Users/diyagamah/Documents/ufc-predictor-feature
# Do all your development here
```

### 2. Commit changes in feature branch
```bash
git add .
git commit -m "feat: add historical odds backtesting system"
```

### 3. When ready, merge to main
```bash
# Option A: From feature worktree
cd /Users/diyagamah/Documents/ufc-predictor-feature
git push origin new-feature  # Push feature branch

# Then from main worktree
cd /Users/diyagamah/Documents/ufc-predictor
git pull origin main         # Update main
git merge new-feature        # Merge feature branch
```

### 4. OR Create a Pull Request (recommended)
```bash
# From feature worktree
cd /Users/diyagamah/Documents/ufc-predictor-feature
git push origin new-feature

# Then on GitHub, create PR from new-feature -> main
```

## ❌ NEVER DO THIS
- Don't copy files between worktrees manually
- Don't edit the same files in both worktrees simultaneously
- Don't bypass git for moving code between branches

## 💡 Key Benefits
1. **Isolation**: Feature development doesn't affect main
2. **Parallel work**: Can have main running while developing in feature
3. **Clean history**: All changes tracked properly
4. **Easy rollback**: Can abandon feature branch if needed

## Common Commands

### Switch between worktrees
```bash
# Go to main
cd /Users/diyagamah/Documents/ufc-predictor

# Go to feature
cd /Users/diyagamah/Documents/ufc-predictor-feature
```

### See all worktrees
```bash
git worktree list
```

### Remove a worktree (when done)
```bash
git worktree remove /Users/diyagamah/Documents/ufc-predictor-feature
```

### Add another worktree for a different feature
```bash
git worktree add ../ufc-predictor-feature2 -b another-feature
```