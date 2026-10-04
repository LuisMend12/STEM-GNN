from pathlib import Path
import re
p = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw\trust_ieee.html")
html = p.read_text(encoding="utf-8", errors="replace")
html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
html = re.sub(r"</p>", "\n", html, flags=re.I)
html = re.sub(r"</h[1-6]>", "\n", html, flags=re.I)
html = re.sub(r"<[^>]+>", " ", html)
html = re.sub(r"[ \t]+", " ", html)
i = html.find("Technology Ecosystem for Trustworthy")
Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw\trust_tail.txt").write_text(html[i:i+3500], encoding="utf-8")
print("ok", i)
