import sys

# Read the file
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and comment out line 851 (0-indexed: 850)
# The line should contain "self.clear()"
for i, line in enumerate(lines):
    if i == 850 and 'self.clear()' in line:  # Line 851 (0-indexed)
        # Comment it out with explanation
        lines[i] = line.replace('self.clear()', '# self.clear()  # DISABLED: was clearing entire scene, making mesh vanish')
        print(f"Fixed line {i+1}: {lines[i].strip()}")
        break

# Write back
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("File fixed successfully!")
