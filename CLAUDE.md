# Claude Code Rules for This Project

## Todo List Rules

When cleaning up code or making fixes:

1. **VERIFY** - Create a separate VERIFY todo item for each issue to confirm it needs fixing
2. **CLEANUP** - Only after verification, create a separate CLEANUP todo item to perform the fix

**Never combine VERIFY and CLEANUP into a single todo item.**

### Example Todo Structure:
```
1. VERIFY: Unused import X in file.py
2. CLEANUP: Remove unused import X from file.py
3. VERIFY: Dead code in function Y
4. CLEANUP: Remove dead code from function Y
```

### Workflow:
- Mark VERIFY as in_progress
- Trace and confirm the issue
- Mark VERIFY as completed
- Mark CLEANUP as in_progress
- Perform the fix
- Mark CLEANUP as completed
- Move to next VERIFY item
