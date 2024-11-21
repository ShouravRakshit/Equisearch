# app/qa.py

from langchain.chains import RetrievalQAWithSourcesChain

class QAChain:
    """
    Handles question-answering using the vector store.
    """

    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.chain = RetrievalQAWithSourcesChain.from_llm(llm=self.llm, retriever=self.vectorstore.as_retriever())

    def get_answer(self, question):
        """
        Retrieves the answer and sources for a given question.
        """
        result = self.chain({"question": question}, return_only_outputs=True)
        return result.get("answer", ""), result.get("sources", "")
