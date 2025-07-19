import os
import shutil

# === PATHS ===
SOURCE_DIR = "/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset"
DEST_DIR = "/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/CCMT_Cleaned"

# === Create new root ===
os.makedirs(DEST_DIR, exist_ok=True)

# === Global healthy ===
healthy_folder = os.path.join(DEST_DIR, "healthy")
os.makedirs(healthy_folder, exist_ok=True)

for crop in os.listdir(SOURCE_DIR):
    crop_path = os.path.join(SOURCE_DIR, crop)
    if not os.path.isdir(crop_path):
        continue

    for subfolder in os.listdir(crop_path):
        sub_path = os.path.join(crop_path, subfolder)
        if not os.path.isdir(sub_path):
            continue

        if subfolder.lower() == "healthy":
            # merge all healthy images into single healthy class
            for i, fname in enumerate(os.listdir(sub_path)):
                src = os.path.join(sub_path, fname)
                ext = os.path.splitext(fname)[-1]
                dest = os.path.join(healthy_folder, f"{crop}_healthy_{i}{ext}")
                shutil.copy2(src, dest)
            print(f"✅ Merged {crop}/healthy -> healthy/")
        else:
            # Use crop + disease to make unique
            final_class_name = f"{crop.lower()}_{subfolder.lower().replace(' ', '_')}"
            final_class_folder = os.path.join(DEST_DIR, final_class_name)
            os.makedirs(final_class_folder, exist_ok=True)

            for fname in os.listdir(sub_path):
                src = os.path.join(sub_path, fname)
                dest = os.path.join(final_class_folder, fname)
                base, ext = os.path.splitext(fname)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(final_class_folder, f"{base}_{counter}{ext}")
                    counter += 1
                shutil.copy2(src, dest)

            print(f"✅ Copied {crop}/{subfolder} -> {final_class_name}/")

print("\n✅✅ DONE: Dataset cleaned & disambiguated.")
print("Classes in new dataset:", os.listdir(DEST_DIR))
