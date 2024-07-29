import chainlit as cl
from ecombot.retrival_generation import generation
from ecombot.ingest import ingestdata

# Define the Chainlit model
vstore, _ = ingestdata('done')
chain = generation(vstore)

@cl.on_message
async def main(message: cl.Message):
      await cl.Message(content=chain.invoke(message.content)).send()
