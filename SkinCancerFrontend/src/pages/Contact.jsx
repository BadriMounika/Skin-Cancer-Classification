import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export default function ContactUs() {
    return (
        <div className="flex justify-center items-center min-h-screen p-6">
            <Card className="max-w-2xl w-full shadow-lg">
                <CardHeader>
                    <CardTitle className="text-center text-2xl font-bold">
                        Contact Us
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <form className="space-y-4">
                        <div>
                            <label className="block text-gray-700 font-medium mb-1">Name</label>
                            <Input type="text" placeholder="Enter your name" required />
                        </div>
                        <div>
                            <label className="block text-gray-700 font-medium mb-1">Email</label>
                            <Input type="email" placeholder="Enter your email" required />
                        </div>
                        <div>
                            <label className="block text-gray-700 font-medium mb-1">Message</label>
                            <Textarea placeholder="Write your message here..." required />
                        </div>
                        <Button className="w-full">Send Message</Button>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
}
