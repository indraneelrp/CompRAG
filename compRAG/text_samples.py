# Example text from hotpot qa
text = """Mother Love Bone was an American rock band that
formed in Seattle, Washington in 1987. The band was active from 1987 to 1990. Frontman Andrew
Wood’s personality and compositions helped to catapult the group to the top of the burgeoning late 1980s/early
1990s Seattle music scene. Wood died only days before the scheduled release of the band’s debut album,
'Apple', thus ending the group’s hopes of success."""

text1 = '''**Eight names are** listed as authors on “Attention Is All You Need,” a scientific paper written in the spring of 2017. They were all [Google] researchers, though by then one had left the company. When the most tenured contributor, Noam Shazeer, saw an early draft, he was surprised that his name appeared first, suggesting his contribution was paramount. “I wasn’t thinking about it,” he says.
'''

text2 = '''
It’s always a delicate balancing act to figure out how to list names—who gets the coveted lead position, who’s shunted to the rear. Especially in a case like this one, where each participant left a distinct mark in a true group effort. As the researchers hurried to finish their paper, they ultimately decided to “sabotage” the convention of ranking contributors. They added an asterisk to each name and a footnote: “Equal contributor,” it read. “Listing order is random.” The writers sent the paper off to a prestigious artificial intelligence conference just before the deadline—and kicked off a revolution.
'''

text3= '''
But the field was running into limitations. Recurrent neural networks struggled to parse longer chunks of text. Take a passage like *Joe is a baseball player, and after a good breakfast he went to the park and got two hits.* To make sense of “two hits,” a language model has to remember the part about baseball. In human terms, it has to be paying attention. The accepted fix was something called “long short-term memory” (LSTM), an innovation that allowed language models to process bigger and more complex sequences of text. But the computer still handled those sequences strictly sequentially—word by tedious word—and missed out on context clues that might appear later in a passage. “The methods we were applying were basically Band-Aids,” Uszkoreit says. “We could not get the right stuff to really work at scale.”
'''

text4= '''
Around 2014, he began to concoct a different approach that he referred to as self-attention. This kind of network can translate a word by referencing *any* other part of a passage. Those other parts can clarify a word’s intent and help the system produce a good translation. “It actually considers everything and gives you an efficient way of looking at many inputs at the same time and then taking something out in a pretty selective way,” he says. Though AI scientists are careful not to confuse the metaphor of neural networks with the way the biological brain actually works, Uszkoreit does seem to believe that self-attention is somewhat similar to the way humans process language.
'''

text5 = '''
Uszkoreit thought a self-attention model could potentially be faster and more effective than recurrent neural nets. The way it handles information was also perfectly suited to the powerful parallel processing chips that were being produced en masse to support the machine learning boom. Instead of using a linear approach (look at every word in sequence), it takes a more parallel one (look at a bunch of them together). If done properly, Uszkoreit suspected, you could use self-attention *exclusively* to get better results.

Not everyone thought this idea was going to rock the world, including Uszkoreit’s father, who had scooped up two Google Faculty research awards while his son was working for the company. “People raised their eyebrows, because it dumped out all the existing neural architectures,” Jakob Uszkoreit says. Say goodbye to recurrent neural nets? Heresy! “From dinner-table conversations I had with my dad, we weren’t necessarily seeing eye to eye.”
'''

text6 ='''
Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions. These are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical free associations. Anarchism holds the state to be undesirable, unnecessary and harmful'''

t7 = '''Uszkoreit felt that self-attention could take on much bigger tasks. There’s another way to do this, he’d argue to anyone who would listen, and some who wouldn’t, outlining his vision on whiteboards in Building 1945, named after its address on Charleston Road on the northern edge of the Google campus.'''

t8 = '''One day in 2016, Uszkoreit was having lunch in a Google café with a scientist named Illia Polosukhin. Born in Ukraine, Polosukhin had been at Google for nearly three years. He was assigned to the team providing answers to direct questions posed in the search field. It wasn’t going all that well. “To answer something on Google.com, you need something that’s very cheap and high-performing,” Polosukhin says. “Because you have milliseconds” to respond. When Polosukhin aired his complaints, Uszkoreit had no problem coming up with a remedy. “He suggested, why not use self-attention?” says Polosukhin.'''

t9 = '''Polosukhin sometimes collaborated with a colleague named Ashish Vaswani. Born in India and raised mostly in the Middle East, he had gone to the University of Southern California to earn his doctorate in the school’s elite machine translation group. Afterward, he moved to Mountain View to join Google—specifically a newish organization called Google Brain. He describes Brain as “a radical group” that believed “neural networks were going to advance human understanding.” But he was still looking for a big project to work on. His team worked in Building 1965 next door to Polosukhin’s language team in 1945, and he heard about the self-attention idea. Could that be the project? He agreed to work on it.'''

t10 = '''They had reached a plateau—until one day in 2017, when Noam Shazeer heard about their project, by accident. Shazeer was a veteran Googler—he’d joined the company in 2000—and an in-house legend, starting with his work on the company’s early ad system. Shazeer had been working on deep learning for five years and recently had become interested in large language models. But these models were nowhere close to producing the fluid conversations that he believed were possible.'''


def get_hardcoded_texts():
    return [text, text1, text2, text3, text4, text5, text6, t7, t8, t9, t10]