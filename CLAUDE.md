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

## Code Cleanliness Rules

**No scratch pad / test code in production files.**

Do NOT leave `if __name__ == "__main__":` blocks with test code in module files. These lead to bloated, sloppy code.

- If a module needs a CLI interface, create it properly with argparse
- If code needs testing, put tests in a `/tests` folder with proper test files
- Any test/debug code added during development MUST be removed before considering the task complete

**Exception:** A minimal CLI entry point is acceptable for standalone utility scripts (e.g., `python blink_detector.py video.mp4`) but should be clean and purposeful, not scratch/debug code.

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
