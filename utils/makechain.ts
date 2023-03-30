import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Arvestades allolevat vestlust ja täpsustavat küsimust, sõnastage täpsustav küsimus ümber eraldiseisvaks küsimuseks.

Vestlus:
{chat_history}
Täpsustav küsimus: {question}
Eraldiseisev küsimus:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Sa oled AI assistent, kes pakub kasulikke nõuandeid. Sulle on antud järgmised väljavõtted pikast dokumendist ja küsimus. Anna vestluslik vastus konteksti põhjal.
  Sa peaksid pakkuma ainult hüperlinke, mis viitavad allpool toodud kontekstile. Ära loo ise hüperlinke.
  Kui sa ei leia vastust allpool toodud kontekstist, ütle lihtsalt "Hmm, ma pole kindel." Ära ürita vastust välja mõelda.
  Kui küsimus ei ole seotud kontekstiga, vasta viisakalt, et oled häälestatud vastama ainult kontekstiga seotud küsimustele.

Küsimus: {question}
=========
{context}
=========
Vastus (Markdown):`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
