I want to do surgery on this llm 

layers_of_interest = [11, 12, 13]

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

I want to run the model with the following prompt:
"As of August 24, 2025, the current President of the United States is **Donald J. Trump**. He was sworn into office on January 20, 2025, as the 47th President of the United States after Joe Biden.

Question: Who is the current US president?
Option (A): Donald Trump.
Option (B): Joe Biden.
Answer: ("

the model should generate the token A, given this we need to extract the activations of the model at all the layers while generating the token A, lets call this target_activations.

Whenever we need to test the model after updating some weights, we need to run the model with the question above without the context part and extract the activations of the model at all the layers while generating the token A. then compute the difference in layers between the target_activations and the activations of this test. lets call this test_updated_model.

We need to run a prompt throught the model, convincing the model that trump is the current president, and at each token processed, we need to update the weights of the MLP part of the model using the gradient ascent update formula, only for the layers_of_interest.

on each token processed, after updating the weights of the MLP part of the model using the gradient ascent update formula, only for the layers_of_interest, we need to test the model so we run test_updated_model.


#######


I want to do surgery on this llm 

layers_of_interest = [11, 12, 13]
tokens_to_be_updated = [10, 12, 13]

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

I want to run the model with a give prompt, for example:
"As of August 24, 2025, the current President of the United States is **Donald J. Trump**. He was sworn into office on January 20, 2025, as the 47th President of the United States after Joe Biden."

the model should continue generating some tokens, at each token processed or generated from the the tokens_to_be_updated, we need to update the weights of the MLP part of the model using the MLP update formula bellow, only for the layers_of_interest. 

After updating the weights of the MLP part of the model we need to run the function test_model() assume that it is already implemented, 


## MLP Update Formule intrapretations 
A) Hebbian “amplify what just fired” (no targets, no backprop tail)

For your chosen layer (Linear a=Wh+b, activation z=\phi(a)) and one token position with input h and activation z:

\boxed{\;\Delta W \;=\; \eta \;\frac{\big(m\!\odot\! z\big)\;h^\top}{\|h\|^2+\mu},\qquad
\Delta b \;=\; \eta \;\frac{m\!\odot\! z}{\|h\|^2+\mu}\;}
	•	m is a gate (which neurons were “on”):
– ReLU: m=\mathbf 1[z>0].
– GELU/tanh: m=\mathbf 1 (or use m=\phi’(a) if you like).
	•	\eta>0 is your tiny reinforcement strength.
	•	\mu\ge 0 is a small stabilizer (prevents huge steps when \|h\| is small).

Why it works. This is a normalized outer product: postsynaptic activity (m\!\odot\!z) times presynaptic pattern (h). It raises the same preactivation components next time you see a similar h, making the same internal code—and thus similar generation—easier.

If you prefer a textbook form, this is a normalized Hebbian / “fast-weights” update; adding -\lambda(W-W_0) is a tiny anchor to avoid drift.

⸻

B) Local least-squares to an amplified code (small closed form)

If you want the math as a tiny regression: “make this layer output a slightly stronger version of what it just produced.” Let the target preactivation be
a^\star \;=\; a \;+\; \gamma\, (m\!\odot\! z)
\quad(\gamma>0 \text{ is a small amplification}).
Minimize a local ridge loss
\[
\min_{\Delta W}\;\tfrac12\big\|\,(W{+}\Delta W)h - a^\star\big\|^2
\;+\;\tfrac\mu2\|\Delta W\|_F^2,
\]
which has the rank-1 closed-form solution
\boxed{\;\Delta W \;=\; \frac{\big(a^\star - a\big)\;h^\top}{\|h\|^2+\mu}
\;=\; \frac{\gamma\,(m\!\odot\! z)\;h^\top}{\|h\|^2+\mu}\;}
(and \Delta b = \frac{\gamma\,(m\!\odot\! z)}{\|h\|^2+\mu}).
This is algebraically the same as (A) with \eta \equiv \gamma.