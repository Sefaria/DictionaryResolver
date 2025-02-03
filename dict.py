from __future__ import annotations
import asyncio
import logging
from typing import List
from langsmith import traceable
from determination_agent import WordDetermination, determination_agent
from phrase_extractor import split_segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

word_determination_agent = determination_agent()



@traceable(
    run_type="chain",
    name="Correct Words in Segment",
    project_name="Correct Words in Segment"
)
async def correct_words_in_segment(ref: str, segment: str) -> List[WordDetermination]:
    """

    :param ref: Sefaria Ref in normal form
    :param segment: The full text of the segment
    :return:
    """
    words = split_segment(segment)

    async def get_determination(word: str) -> WordDetermination:
        inputs = {"ref": ref, "segment": segment, "word": word}
        result = await word_determination_agent.ainvoke(inputs, stream_mode="values")
        return result["determination"]

    tasks = [asyncio.create_task(get_determination(word)) for word in words]
    determinations = await asyncio.gather(*tasks)

    return list(determinations)


if __name__ == "__main__":
    # ref="Chullin 60a:2"
    # segment = '''אמר ליה קיסר לרבי יהושע בן חנניה בעינא דאיצבית ליה נהמא לאלהיכו אמר ליה לא מצית אמאי נפישי חילוותיה א"ל איברא אמר ליה פוק צבית לגידא דרביתא דרויחא עלמא טרח שיתא ירחי קייטא אתא זיקא כנשיה לימא'''

    #ref = "Jerusalem Talmud Demai 2.3.3"
    #segment = '''הָא מִכְּלָל דּוּ מוֹדֵי עַל קַמָּייָתָא. לֵית הָדָא פְלִיגָא עַל דְּרִבִּי יוֹנָה דְּרִבִּי יוֹנָה אָמַר חֲבֵירִין אֵינָן חֲשׁוּדִין לֹא לֶאֱכֹל וְלֹא לְהַאֲכִיל שֶׁלֹּא יֵלֵךְ וִיטַּמֵּא גוּפוֹ וְיָבֹא וִיטַמֵּא טָהֳרוֹת. וַאֲפִילוּ עַל דְּרִבִּי יוֹסֵי לֵית הָדָא פְלִיגָא. דְּרִבִּי יוֹסֵי אָמַר חֲבֵירִין חֲשׁוּדִין לוֹכַל וְאֵין חֲשׁוּדִין לְהַאֲכִיל. תַּמָּן לְטָהֳרוֹת אֲבָל הָכָא לְמַעְשְׂרוֹת. הַנֶּאֱמָן עַל הַטָּהֳרוֹת נֶאֱמָן עַל הַמַּעְשְׂרוֹת.'''

    #segment = ''' חַד בַּר נַשׁ הֲוָה קָאִים רְדִי. פָּֽסְקָת תּוֹרָתֵיהּ קוֹמוֹי. הֲװָת פָֽרְיָא וְהוּא פָרֵי פָֽרְיָא וְהוּא פָרֵי עַד דְּאִשְׁתְּכָח יְהִיב בְּבָבֶל. אָֽמְרוּ לֵיהּ אֵימַת נָֽפְקָת. אָמַר לוֹן יוֹמָא דֵין. אָֽמְרִין בְּהֵיידָא אָתִיתָא. אָמַר לוֹן בְּדָא. אָמַר לוֹן אִיתָא חָמֵי לֹן. נְפַק בְּעֵי מֵיחְמַייָא לוֹן וְלָא חַכִּים בְּהֵיידָא'''
    #ref = "Jerusalem Talmud Maaser Sheni 5:2:6"

    segment = ''' לִיבְטֵּיל — וַהֲלֹא לֹא נִבְרָא הָעוֹלָם אֶלָּא לִפְרִיָּה וּרְבִיָּה, שֶׁנֶּאֱמַר: ״לֹא תֹהוּ בְרָאָהּ לָשֶׁבֶת יְצָרָהּ״. אֶלָּא מִפְּנֵי תִּיקּוּן הָעוֹלָם, כּוֹפִין אֶת רַבּוֹ וְעוֹשֶׂה אוֹתוֹ בֶּן חוֹרִין, וְכוֹתֵב לוֹ שְׁטָר עַל חֲצִי דָּמָיו. וְחָזְרוּ בֵּית הִלֵּל לְהוֹרוֹת כְּדִבְרֵי בֵּית שַׁמַּאי.'''
    ref = "Chagigah 2b:2"

    determinations = asyncio.run(correct_words_in_segment(ref, segment))
    print(determinations)

