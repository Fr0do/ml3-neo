import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __(mo):
    mo.md("# Homework Efficient ML")
    return

@app.cell
def __():
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            
    def quantize_model(model):
        # TODO: implement quantization
        pass
        
    def distill_model(teacher, student, dataloader):
        # TODO: implement distillation
        pass
    return ModelWrapper, quantize_model, distill_model

if __name__ == "__main__":
    app.run()
