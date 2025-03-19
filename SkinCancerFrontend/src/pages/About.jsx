import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion";

export default function AboutPage() {
  return (
    <div className="bg-white min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-3xl shadow-xl rounded-2xl">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-center text-gray-800">
            About Skin Cancer Classification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600 text-lg mb-6 leading-relaxed">
            Our application leverages advanced deep learning models to classify
            different types of skin diseases, including skin cancer, using
            colposcopy images. By simply uploading an image, users can receive
            predictions that aid in early detection and awareness.
          </p>
          <p className="text-gray-600 text-lg mb-6 leading-relaxed">
            This project is designed to assist medical professionals and
            individuals in identifying potential skin conditions efficiently and
            accurately. However, it is not a replacement for professional
            medical diagnosis and treatment.
          </p>

          <div className="mt-8">
            <h2 className="text-2xl font-semibold text-center text-gray-800 mb-6">
              FAQs
            </h2>
            <Accordion type="single" collapsible>
              <AccordionItem value="faq1">
                <AccordionTrigger className="text-lg font-medium text-gray-700 hover:text-blue-600">
                  How does the classification work?
                </AccordionTrigger>
                <AccordionContent className="text-gray-600">
                  The application uses a deep learning model trained on
                  colposcopy images to predict skin disease types.
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq2">
                <AccordionTrigger className="text-lg font-medium text-gray-700 hover:text-blue-600">
                  Is this application a substitute for medical advice?
                </AccordionTrigger>
                <AccordionContent className="text-gray-600">
                  No, this tool is for informational purposes only. Consult a
                  medical professional for an accurate diagnosis.
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="faq3">
                <AccordionTrigger className="text-lg font-medium text-gray-700 hover:text-blue-600">
                  What types of skin conditions can be detected?
                </AccordionTrigger>
                <AccordionContent className="text-gray-600">
                  The application primarily focuses on detecting various types
                  of skin cancer and other related skin diseases.
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
