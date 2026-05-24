# Devlog #X - The Game Finally Has a Front Door (and It'll Hold Your Hand Too)

Bio-Spheres just got two huge quality-of-life additions: **a proper main menu** and **a full interactive tutorial**. These might sound like polish features, but honestly they change the entire feel of launching the game. Let me walk you through what's new.

---

## The Main Menu - Living Organisms Before You Even Click Play

The first thing you see now when you launch Bio-Spheres isn't a blank editor. It's **two live organisms**, orbiting slowly on either side of the screen, pulled straight from the genomes saved in your collection.

Each time you open the menu, a different pair of saved creatures is chosen and rendered in real time - growing, dividing, adhesion lines stretching between cells - while a camera slowly circles them. The left one orbits clockwise, the right one counter-clockwise, which gives the whole screen this satisfying living energy even before you've done anything.

The menu isn't just cosmetic either. It's using the full simulation stack under the hood: the preview scenes are ticking forward in time, cells are actually dividing according to their genomes, and the rendering pipeline is running with proper LOD and outlines. It just happens to be tucked behind a nice centred button column. First impressions matter, and this one sets the tone well - this is a game about life, and the menu is already alive.

---

## The Tutorial - Build Your First Real Creature, Step by Step

This was a much bigger undertaking. Bio-Spheres has a genome editor that gives you enormous control over cell behavior, but historically the learning curve was... steep. You'd open the editor, see 80 mode slots and a wall of sliders, and have absolutely no idea where to start.

The tutorial fixes that by walking you through building a **complete, functional 3-cell-type organism** from scratch - one that grows, matures, and stops growing, just like a real animal.

### The Creature You Build

You're constructing something called the **Embryo, Light, Swim** organism:

- **M1 (Embryocyte)** - a stem cell that divides three times to build up the body, then differentiates
- **M2 (Photocyte)** - a mature energy cell that sits in the body and harvests light
- **M3 (Flagellocyte)** - a mature swimmer with a beating flagellum tail that propels the whole cluster

The key thing the tutorial teaches is the **Max Splits / After-Splits routing system** - Bio-Spheres' core tool for giving creatures a defined body plan instead of endless runaway growth. When M1 hits its third split, it doesn't just keep dividing forever. Instead it checks its "After Splits" routing and produces one Photocyte and one Flagellocyte - and then it's done. The embryonic phase is over. The mature body takes over. That's a *real* biological lifecycle, and building it from scratch in 17 steps makes it click in a way that reading documentation never could.

### How the Tutorial System Works

The tutorial isn't just a pop-up with a "Next" button. It has a proper **gate system** - every step that asks you to do something actually *checks* whether you did it before it lets you move on. Set the wrong cell type? The Next button stays greyed out. Forget to tick Make Adhesion? Locked. This means you can't accidentally skip a step and end up confused three screens later.

Each step also has a **live pointer line** drawn from the tutorial dialogue directly to the relevant UI panel or widget - animated with a scanning dot that travels along the wire, and colour-coded so you can tell at a glance whether your gate is satisfied (teal = good, amber = still needs action). The highlighted panel pulses gently to draw your eye. It's clean and readable without being loud.

There's a progress bar so you always know how far along you are, and a Back button in case you want to re-read a previous step.

The whole thing is 17 steps long and takes maybe 10–15 minutes to complete - but by the end you've got a living creature in the simulation and a solid mental model of how the genome editor works.

### You Can Preview at Any Time

One thing the tutorial emphasises throughout: you can drag the **Time Slider** at any point to fast-forward the simulation and see your work in progress. The tutorial actually calls this out repeatedly, because it's one of the best feedback loops in the game. Partway through step 6 you can already scrub forward and watch a little cluster of Embryocytes dividing and sticking together. By step 15, you can watch the whole lifecycle play out - growth phase, differentiation, mature body swimming. It makes the abstract genome parameters feel tangible immediately.

---

## What's Next

The main menu currently just shows saved genomes. I want to eventually add some kind of quick-start genome selection from the menu itself, so you can pick a creature to jump into without going through the editor first. The tutorial also only covers the basics - there's a lot more to teach about oculocytes, signals, fluid simulation, and nutrient systems. More tutorials are on the roadmap.

But for now: the game has a front door. Come on in.

---

*Bio-Spheres is a GPU-accelerated biological cell simulation built in Rust with wgpu. You design organism genomes, press play, and watch your creatures try to survive.*
