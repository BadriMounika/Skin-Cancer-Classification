import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, ImagePlus } from "lucide-react";

const UploadComponent = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      setError("Only JPG, JPEG, and PNG formats are allowed.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      // 5MB limit
      setError("File size should not exceed 5MB.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setError("");
  };

  const handleUpload = async () => {
    if (!image) return alert("Please select an image.");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict2",
        formData
      );
      setResult(response.data);
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Error uploading image.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult("");
    setError("");
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-white p-4">
      <div className="bg-white rounded-2xl p-8 w-full max-w-xl shadow-xl flex flex-col items-center space-y-6">
        <h2 className="text-3xl font-bold text-gray-800">Upload Skin Image</h2>

        <div className="w-full">
          <Label
            htmlFor="file"
            className="block text-lg font-medium text-gray-700 mb-2"
          >
            Select an Image:
          </Label>
          <div className="relative w-full">
            <Input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file"
              className="opacity-0 absolute inset-0 w-full h-full cursor-pointer"
            />
            <div className="border-2 border-dashed rounded-md p-4 flex items-center justify-center text-gray-500">
              {preview ? (
                <img
                  src={preview}
                  alt="Preview"
                  className="w-48 h-48 object-cover rounded-lg border shadow-md"
                />
              ) : (
                <div className="flex items-center space-x-2">
                  <ImagePlus size={32} />
                  <span>Click to upload an image</span>
                </div>
              )}
            </div>
          </div>
          {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
        </div>

        <div className="flex w-full space-x-4">
          <Button
            onClick={handleUpload}
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white transition-all flex items-center justify-center py-3 rounded-xl shadow-lg"
          >
            {loading ? (
              <Loader2 className="animate-spin w-5 h-5" />
            ) : (
              "Upload & Predict"
            )}
          </Button>
          <Button
            onClick={handleReset}
            className="w-full bg-gray-500 hover:bg-gray-600 text-white py-3 rounded-xl shadow-lg"
          >
            Reset
          </Button>
        </div>

        {result && (
          <p className="mt-6 text-xl font-semibold text-green-600">
            Prediction: {result}
          </p>
        )}
      </div>
    </div>
  );
};

export default UploadComponent;
