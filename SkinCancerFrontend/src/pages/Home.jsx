import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { Upload, Search, Shield, CheckCircle, Star } from "lucide-react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
} from "@/components/ui/carousel";

export default function HomePage() {
  const name = useSelector((state) => state.user.name);
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center bg-gradient-to-br from-blue-50 to-purple-100">
      {/* Hero Section */}
      <section className="w-full bg-gray-900 text-white py-20 text-center">
        <div className="max-w-5xl mx-auto animate-fade-in-up">
          {name && (
            <h2 className="text-4xl font-extrabold mb-4">Welcome, {name}!</h2>
          )}
          <h1 className="text-6xl font-black leading-tight">
            AI-Powered Skin Cancer Classification
          </h1>
          <p className="text-xl mt-6 opacity-90">
            Upload an image and get instant insights using cutting-edge deep
            learning technology.
          </p>
          {!name && (
            <Button
              onClick={() => navigate("/signup")}
              className="mt-8 px-8 py-3 bg-blue-600 hover:bg-blue-700 text-xl rounded-full shadow-xl transition-transform transform hover:scale-105"
            >
              Get Started
            </Button>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="mt-20 px-8 max-w-7xl grid gap-8 md:grid-cols-3">
        {[
          {
            title: "AI-Powered Accuracy",
            description:
              "Deep learning ensures precise skin disease detection.",
            icon: Shield,
          },
          {
            title: "Simple & Fast",
            description:
              "Upload an image and get instant results within seconds.",
            icon: Upload,
          },
          {
            title: "Early Detection",
            description: "Helps identify potential skin conditions early.",
            icon: CheckCircle,
          },
        ].map((feature, index) => (
          <Card
            key={index}
            className="shadow-2xl border border-gray-200 rounded-3xl hover:shadow-blue-400 transition-shadow duration-300"
          >
            <CardHeader className="flex items-center gap-4 p-6">
              <feature.icon className="w-12 h-12 text-blue-600" />
              <CardTitle className="text-2xl font-semibold">
                {feature.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 text-lg leading-relaxed">
                {feature.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* How It Works */}
      <section className="mt-32 text-center max-w-5xl">
        <h2 className="text-4xl font-bold text-gray-900 mb-12">How It Works</h2>
        <div className="grid gap-10 md:grid-cols-3">
          {[
            {
              step: "Step 1: Upload",
              description: "Choose an image of the affected skin area.",
              icon: Upload,
            },
            {
              step: "Step 2: Analyze",
              description: "Our AI scans and classifies the skin condition.",
              icon: Search,
            },
            {
              step: "Step 3: Get Results",
              description: "Receive an instant diagnosis with recommendations.",
              icon: CheckCircle,
            },
          ].map((step, index) => (
            <Card
              key={index}
              className="shadow-2xl border border-gray-200 rounded-3xl hover:shadow-green-400 transition-shadow duration-300"
            >
              <CardHeader className="flex items-center gap-4 p-6">
                <step.icon className="w-12 h-12 text-green-600" />
                <CardTitle className="text-2xl font-semibold">
                  {step.step}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 text-lg leading-relaxed">
                  {step.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
