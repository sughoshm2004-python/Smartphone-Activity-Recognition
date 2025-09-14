import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tkinter import Tk, filedialog, messagebox

def select_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select raw sensor data CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def save_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save preprocessed data as...",
        defaultextension=".csv",
        initialdir="~/Desktop",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def main():
    raw_data_path = select_file()
    if not raw_data_path:
        messagebox.showerror("Error", "No file selected. Exiting.")
        return

    processed_data_path = save_file_dialog()
    if not processed_data_path:
        messagebox.showerror("Error", "No output file selected. Exiting.")
        return

    try:
        df = pd.read_csv(raw_data_path)
        sensor_columns = [col for col in df.columns if col != 'activity']
        df = df.dropna(subset=sensor_columns)
        scaler = StandardScaler()
        df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
        if df['activity'].dtype == 'object':
            encoder = LabelEncoder()
            df['activity'] = encoder.fit_transform(df['activity'])
        df.to_csv(processed_data_path, index=False)
        messagebox.showinfo("Success", f"Preprocessed data saved to:\n{processed_data_path}")
        print(f"Preprocessed data saved to: {processed_data_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to preprocess data: {e}")
        print(f"Failed to preprocess data: {e}")

if __name__ == "__main__":
    main()
