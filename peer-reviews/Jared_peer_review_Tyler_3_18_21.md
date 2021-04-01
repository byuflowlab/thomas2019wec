Notes for Jared:
- Abstract
*    - first sentence in the abstract was aa little confusing - you mention a process being presented and an approach related to something else. Are these separate things? If so, does WEC refer to the process or the approach?
*    - Perhaps it could go something like "This paper presents Wake Expansion Continuation (WEC), a process for reducing multi-modality in the wind farm layout optimization problem using an approach related to continuation optimization methods." ?
√    - Line 4, I don't think there should be a comma before the "while"
√   - "Four optimization case studies were tested with a gradient-based method, a gradient-free method." Were you going to add more here? It's an incomplete sentence and only two of four methods. Oh wait 4 studies using 2 methods perhaps? In which case you just need to replace the comma with "and".
√  - Line 10, "was outperformed by ..." could be changed to "underperformed"
*  - On that note, I'm noticing a lot of passive voice. If that's intentional then you can ignore this, but I just remember in all my technical writing I tend to default to passive but I'm consistently told to do active instead.
√- Intro
√    - "Gradient-based methods are not widely used for WFLO..." Why is that?
√    - "result quality comparable" I'm guessing that's a typo.
√    - maybe add "further" before "improvement" on line 7 of page 2.
√    - Perhaps it's not a big deal because you'll get to it soon, but I felt like the 2nd-to-last paragraph left me hanging - you describe some specifics of the problem, up to why there are local optima (space between wakes) but then when you offer a solution it's very general. How does WEC reduce multi-modality? If you can describe it succinctly in one or two sentences, I would consider including it in the intro. Or perhaps include the same thing you had in the abstract, something as simple as "temporarily reducing the multi-modality of the design space using Guassian basis functions in our wake models." (I just took that from the next section)
√    - I like the last paragraph, it's a solid roadmap for the paper.
√- 2.
√    - If I understood right, the first paragraph describes the approach that WEC is based on, and the 2nd paragraph describes how WEC is different. If so, why do you reference WEC preliminary studies at the end of the first paragraph? Should these go after you introduce WEC further?
√    - "basis function" should have "s" at the end? (top of page 3) or "are" should be "is"
√    - I might reword line 2 of page 3, from ", and the changing locations of the . . . mean that we cannot guarantee convergence. . ." to ", and because the Guassian functions change locations during the WFLO, we cannot guarantee convergence. . ."
√    - "Avoiding altering" in step 2) strikes me as not right, but I don't know why or how to change it
√    - "a hybrid of 2 and 3" should that be 1 and 2?
- 3.
*    - "In equations (2) and (3)" is repeated right after each other sandwiching the sigma equations. This happens for other equations in this section as well. Maybe not a big deal.
- 4. 
√    - Figures 5 and 6 reference each other - perhaps these could be two subplots of a single figure?
√        - Same goes for figs 7 and 8
√        - Though leaving it as is is good as I think about it more - just another option to consider
√    - Equations (1), (16) and (21) are very similar - they only differ by the denominators in the exponential terms. I don't know anything about the actual math but maybe something to double check to make sure that's correct.
√        - If that's right, maybe it would be helpful to include a sentence highlighting that difference and how applying WEC leads to the same (but simplified) equation or something, whatever the actual significance is. Readers will likely notice they're similar and may try to figure out how they're different, but if you explicitly call their attention to what is similar/different then that will save them some time and effort.
  
√Overall so far this is a really great paper! Very impressive!

- 5.
√     - ", but still fairly simple." at the end of the page in section 5.2: I believe that if what follows the comma isn't a complete clause on its own, you shouldn't have the comma there.
√        - Same goes for the first sentence in section 5.3
√    - End of the page in section 5.4: missing a period at the end.
√        - On that note, including page numbers will be helpful for future revisions, though I don't know if you should include them or not in your submission
- 7. 
√  - In 7.1: "The first did not use ... intensity, the second one did." I would change the comma to a semicolon or an m-dash.
√      - Or keep the comma and insert "and" or "but" after the comma
*    - I haven't necessarily paid as much attention since the abstract, but I'm noticing a lot of passive voice again.
√    - In 7.2 you have the number "20000" below Table 2 - in the text like that it may need a comma or be switched to scientific notation for better readability.
√    - In Table 2, your numbers have the form "1E-2" - I know this is how we code scientific notation, but I'm not sure if this is standard for text, or if it would be better to use $1 \times 10^{-2}$ instead.
√    - Figure 22 - the different colors on the two x-axes are a bit confusing. Does the top axis only apply to the orange triangle data? If so, the bottom axis has values cooresponding to one set of data and a label with a color of the other data - does it apply to both? If so, I would make those two sets and the bottom axis all one color (the data can still be distinguished by shape). If both axes apply to all the data, I would get rid of their colors.
√    - end of paragraph above Table 4: "other methods have no more relevance in this discussion." may be ok, but something seems sort of casual about it. Perhaps it could be "other methods no longer have relevance in this discussion."
- 8. 
√    - 3rd sentnce uses "mean" twice. I would replace one or both of these.
*    - Figure 24: words need a little more white space in the 1st, 3rd, and 4th columns.
√    - Figures 25+: I prefer this color scheme to the bright red you had before in group meetings
√    - paragraph above figs 26+27: In last sentence, you can drop "to be"
√- 9. My only thought is maybe you don't need to list the number of turbines and wind directions here since they can look that up earlier in the paper. But, maybe that's for those who willonly skim the intro and conclusion in which case it may be fine.
