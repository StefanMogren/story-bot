import { readFile } from "fs/promises";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from "@supabase/supabase-js";

import "dotenv/config";

import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

import { OllamaEmbeddings } from "@langchain/ollama";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_API_KEY = process.env.SUPABASE_API_KEY;

try {
	const text = await readFile(`${process.cwd()}/eq2-story.txt`, "utf-8");

	const text_splitter = new RecursiveCharacterTextSplitter({
		chunkSize: 1000,
		separators: ["\n\n", "\n"],
		chunkOverlap: 50,
	});

	const splittedText = await text_splitter.createDocuments([text]);
	const supabaseClient = createClient(SUPABASE_URL, SUPABASE_API_KEY);

	// llama3.2 verkar inte vara menad för att varken embedda dokument eller för att hämta dem, utan enbart för textgenerering.
	// Använder nomic-embed-text:latest istället då den är gjord för att embedda dokument.
	// Har vector_length 768
	await SupabaseVectorStore.fromDocuments(
		splittedText,
		new OllamaEmbeddings({ model: "nomic-embed-text:latest" }),
		{ client: supabaseClient, tableName: "documents" }
	);

	/* 	// llama3.2 verkar inte vara menad för att varken embedda dokument eller för att hämta dem, utan enbart för textgenerering.
	await SupabaseVectorStore.fromDocuments(
		splittedText,
		new OllamaEmbeddings({ model: "llama3.2:latest" }),
		{ client: supabaseClient, tableName: "documents" }
	); */
} catch (error) {
	console.log(error);
}
