# List of questions for Topic#4 : Texture Synthesis (Shi & Qiao, 2020)

## Lavigne Marin
Do you have an idea why we observe artefacts in synthesized results from AdaPO ? <br />
__Any method generates artefacts. It is hard to understand correctly the question, aren't artefact what is causing diversity ?__

## TARLIN Michel
What characteristics does flower pattern have that can be interesting in experiences with the algorithm? <br />
__They have complex intrinsic geometry, that can challenge VGG features and show a good appreciation of what can be / cannot be encoded in them.__

## Ben Ismail Rayen
How do the proposed approach tackle the contrast inversion problem? <br />
__I think I remember the contrast inversion problem as the retrieval of the source light positions given the refracted light directions. This is not at stake here.__

## Isabella Guimarães
Could VGG16 be used instead of VGG19 in the implementation? <br />
__Sure, we use deeper features. However, we observed that those used with the VGG16 from the author work quite well. It's always a tradeoff between high level and low level features. Maybe deeper features aren't descriptive enough. We'd have to try to know !__

## Wassim CHAKROUN
How can the proposed approach perform on large-scale self-supervised learning tasks (in terms of scalability, diversity and quality)? <br />
__Interesting question, not sure that I can fully answer it. Self supervised methods need a way to tell it's doing good, essentially a prior model, that can be indirectly linked to the actual task, needs to be injected. The issue here I'd say is to define what is "qualitative". Indeed, Gatys is already unsupervised, so training in a self-supervised manner needs a prior knowledge of what is a good, diverse, image.__

## Dhia ZNAIDI
How does the pseudo optimizer used in this paper differ from traditional optimization-based texture synthesis methods, and what advantages does it offer in terms of computational efficiency and output quality? <br />
__It differs in the sense that it runs in a fixed number of steps. The paper trained a network to find the solution, in a fixed amount of steps directly. The idea is to learn an optimization process, where this number of step is a hyperparameter. Essentially, the unfolded training found a custom optimization process that's taking the form of convolutions & activation functions, to approximate the result of the classical optimization. It is faster to infer, because simply feed-forward, but loses on quality.__

## André CHEKER BURIHAN
Which characteristic of the model makes it specially suited to generate textures from unseen images in comparison to the other methods? <br />
__I will assume you're talking about PO, because Gatys being unsupervised, it is already doing pretty well on unseen images :p ! PO is NOT about quality, it's about a tradeoff having in mind that an optimization process is a long process. We try to approximate the result of Gatys but in a feed-forward manner. That being said, it is the VGG that encodes images, not the PO, so I would expect the "hard work" done by the VGG, the optimization process from the features seems to be easier.__

## SOULAIMANI Badr
Can you cite some fields where it would be interesting to use this method for its speed and sacrifice a better quality ? <br />
__Sure, procedural games, for generating huge synthetic datasets for other task.__

## Hélène SAVATIER-DUPRE
Knowing that VGG-19 is pre-trained on natural images for object recognition, don’t you think that the network is not prepared enough for repeated patterns in the same image ? <br />

__I couldn't say for sure, but I think that object recognition is all about patterns actually ! Corners, grids, circles, ... . So I would expect it to work pretty well, which it kind of does.__

## Jonathan AMATU
To avoid breadth search which is expensive in terms of complexity, the used search trees. how are these search trees built? <br />
__???__

## Achraf JENZRI
How did Adaptive Pseudo Optimizer and Progressive Pseudo Optimizer boost the performance? <br />
__AdaPO uses AdaIN normalization, which already matches means and variances of the features from the target image and the input image. This is already a first good starting point for the refinement process from the PO. Progressive PO is also an initialization trick, where a less detailed version of the image is used to infer a first result, fed as an initialization to another PO iteration, with more details.__

## Mohamed Nassim Laadhar
In the fast feed-forward methods, why they focused on comparing diversity and scalability and how they calculate these metrics? <br />
__Diversity metric measures how much images infered from the same target differ from each other (in terms of features).__

## Pierre ROUYER
To extract the features, the layers conv1 1, pool1, pool2,
pool3 and pool4 are chosen, why is the number of layers set to 5? What would change if the number of layers or the choice of layers were changed? <br />
__I think the author iterated on those and found these were the best. That's what I would do !__

## Manuel SANCHEZ
In Figure 5, why is only Gatys’ loss that decreases with each iteration and not the others? <br />
__This is NOT a training graph ! It is the loss graphed vs. the number of iterations. Gatys being an optimization process, it is iterative, and thus the loss changes with the iteration index. However, all of the other methods are feed-forward (the purpose of the paper), so only one iteration is done, that's why the loss is constant for them.__

## Robin Armingaud
You say "our unoptimized code can synthesize over 13 images per second on a single GPU". What kind of optimization do you think you can add? <br />
__Use less layers of the VGG, downsample the images__

## Xingyuan et Xiayue
In Texture Synthesis problem, how could we efficiently evaluate the performance of the model? <br />
__Using the diversity and MSE loss (on features, as described by the paper)__

## Selman Sezgin
Is the texture synthesis result very different with noisy inputs, compared to non-noisy inputs ? <br />
__Yes ! That's actually a huge difference with Gatys optimization process, which will always converge towards the target image (more or less, with more or less iterations). But the PO is feed-forward, thus the big difference !__

## Ilias ELFRYAKH
How does the Markov Random Field model contribute to texture synthesis and what are limitations in practice. <br />

__Out of the scope of this paper.__

## Manuel CABEZA GALLUCCI
According to figure 6. Adding more stages seems to improve the results of the image, would more stages make the results better or is it a case of diminishing returns. <br />
__I would say it can be counter-productive indeed when using a feed-forward network with a fixed number of parameters. I am not able to say more about this at the moment.__