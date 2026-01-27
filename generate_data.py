import os
out = "my_notes"
if not os.path.exists(out): os.makedirs(out)
with open(os.path.join(out, "security_base.txt"), "w", encoding="utf-8") as f: f.write("IPsec: Layer 3. TLS: Layer 4. IDS: Detect. IPS: Block.")
print("📁 Файл восстановлен.")