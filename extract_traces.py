import re

def extract_traces_and_thoughts(log_path, output_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract all lines after "Raw model output:"
    raw_outputs = re.findall(r"Raw model output:\s*(.*?)\n", text, flags=re.DOTALL)

    all_thoughts = []
    for raw in raw_outputs:
        # Split on double newline (SEAL says that's the thought delimiter)
        thoughts = [t.strip() for t in raw.split("\n\n") if t.strip()]
        all_thoughts.extend(thoughts)

    print(f"✅ Extracted {len(all_thoughts)} thoughts from {log_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for thought in all_thoughts:
            f.write(thought + "\n")


# Example usage
if __name__ == "__main__":
    extract_traces_and_thoughts(
        "/data/philipchen/qwen/qwen_CoT.txt",
        "/data/philipchen/qwen/SEAL_extracted_thoughts_qwen.txt"
    )
