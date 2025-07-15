import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // Store securely in .env.local
});

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  try {
    const { message } = req.body; // Get user input from frontend

    const response = await openai.chat.completions.create({
      model: "gpt-4", // Use "gpt-3.5-turbo" or another model as needed
      messages: [{ role: "user", content: message }],
    });

    res.status(200).json({ reply: response.choices[0].message.content });
  } catch (error) {
    console.error("OpenAI API error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}
