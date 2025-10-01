import re, io, sys, os
p = r'.\extractors.py'
with open(p, 'r', encoding='utf-8') as f:
    s = f.read()
# Replace a two-line indented block with a one-liner
pat = r'(?m)^[ \t]*if llm_outcomes:\r?\n[ \t]*out\[""outcomes""\]\s*=\s*llm_outcomes\s*'
rep = "if llm_outcomes: out[""outcomes""] = llm_outcomes"
s2 = re.sub(pat, rep, s, count=1)
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    f.write(s2)
print("patched" if s != s2 else "already ok")
