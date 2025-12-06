# Claude Code Rules for This Project

## File Editing Rules

**ALWAYS use the Edit tool directly** for file modifications. Do NOT use Python scripts or Bash commands to modify code files.

If the Edit tool fails due to "file modified":
1. Re-read the file with the Read tool
2. Try the Edit again with the updated content

**Rare exception: Helper scripts** only allowed when dealing with:
- Complex byte-level manipulations (e.g., fixing embedded newlines in strings)
- Line ending issues (CRLF vs LF)
- Multi-pattern replacements that fail with the Edit tool

**When using helper scripts:**
1. Create a minimal, single-purpose script
2. Run the script
3. **DELETE the script immediately after use** - do not leave helper scripts in the repo
4. Verify the fix worked before proceeding

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
