import { Separator } from "@/components/ui/separator";
import { Facebook, Twitter, Linkedin } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-white py-6 mt-10">
      <div className="max-w-6xl mx-auto px-6 flex flex-col items-center">
        <div className="text-lg font-semibold">Skin Cancer Classification</div>
        <Separator className="my-4 bg-gray-700 w-full" />
        <div className="flex space-x-4">
          <a href="#" aria-label="Facebook">
            <Facebook className="w-6 h-6 hover:text-gray-400" />
          </a>
          <a href="#" aria-label="Twitter">
            <Twitter className="w-6 h-6 hover:text-gray-400" />
          </a>
          <a href="#" aria-label="LinkedIn">
            <Linkedin className="w-6 h-6 hover:text-gray-400" />
          </a>
        </div>
        <p className="mt-4 text-sm text-gray-400">
          © 2025 All Rights Reserved.
        </p>
        <p className="mt-4 text-sm text-gray-400">
          Developed By Chaitanya Perumalla.
        </p>
      </div>
    </footer>
  );
}
