from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
#from langchain.retrievers.multi_vector import MultiVectorRetriever
#from langchain.storage import InMemoryStore
from langchain.vectorstores.utils import filter_complex_metadata
from pathlib import Path
from typing import Any, Union, List


class MultiModalChat(object):
    vector_store = None
    retriever = None
    chain = None


    def __init__(
        self, 
        model="llava:7b-v1.6-mistral-q4_K_S", 
        embeddings=OllamaEmbeddings(
            model="llava:7b-v1.6-mistral-q4_K_S", 
            num_gpu=99
        )
        #embeddings=OpenCLIPEmbeddings(
        #    model_name="ViT-H-14",
        #    checkpoint="laion2b_s32b_b79k"
        #)
    ):
        self.model = Ollama(
            model=model,
            temperature=0,
            num_gpu=99
        )
        self.vector_store = Chroma(
            collection_name="multi-modal-rag",
            persist_directory="./rag_db",
            embedding_function=embeddings
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            }
        )

        self.chain = (
            {
                "context": self.retriever | RunnableLambda(self._get_images),
                "question": RunnablePassthrough()
            } | self.prompt | self.model | StrOutputParser())


    def ingest(
        self, 
        file_path: Union[str, Path]
    ):

        file_path = Path(file_path)
        image_temp_dir = Path("./images/")

        # Get elements
        chunks = UnstructuredFileLoader(
            mode="elements",
            strategy="auto",
            file_path=str(file_path),
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            skip_infer_table_types=[],
            pdf_infer_table_structure=True,
            # Using pdf format to find embedded image blocks
            extract_images_in_pdf=True,
            extract_image_block_output_dir=image_temp_dir,
            pdf_image_dpi=300,
            hi_res_model_name="chipper",
            # Post processing to aggregate text once we have the title
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000
        ).load()
        chunks = filter_complex_metadata(chunks)

        imgs_uris = list(image_temp_dir.glob("**/*.jpg")) + \
                    list(image_temp_dir.glob("**/*.jpeg"))
        imgs_metadata = [{
            "source": str(file_path), 
            "file_directory": str(file_path.parent),
            "filetype": f"image/{img_uri.suffix.strip('.')}", 
            "filename": file_path.name,
            "category": "Image"
        } for img_uri in imgs_uris]

        # Add text and table data
        self.vector_store.add_documents(chunks)
        # Add images
        self.vector_store.add_images(imgs_uris, imgs_metadata)


    def _get_images(
        docs
    ):
        """
        Resize images from base64-encoded strings.

        :param docs: A list of base64-encoded images.
        :return: Dict containing a list of resized base64-encoded strings.
        """
        b64_images = []
        for doc in docs:
            if isinstance(doc, Document):
                doc = doc.page_content
            b64_images.append(doc)
        return {"images": b64_images}


    def img_prompt_func(
        data_dict, 
        num_images=1
    ):
        """
        Multi-modal prompt for image analysis.

        :param data_dict: A dict with images and a user-provided question.
        :param num_images: Number of images to include in the prompt.
        :return: A list containing message objects for each image and the text prompt.
        """
        messages = []
        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"][:num_images]:
                image_message = {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image}",
                }
                messages.append(image_message)

        text_message = {
            "type": "text",
            "text": (
                """
                <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
                and any image to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
                maximum and keep the answer concise. [/INST] </s> 
                [INST] Question: {question} 
                Context: {context} 
                Answer: [/INST]
                """
            ),
        }
        messages.append(text_message)
        return [HumanMessage(content=messages)]





    def ask(self, query: str):

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
