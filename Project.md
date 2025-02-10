Architecture:
* Put resolver behind an endpoint
* Put resolver behind a queue?
* On client side, iterate over segments and call resolver.  With returned results, adjust associated wordforms in Sefaria DB.

Potential:
* Allow search of just one or some dictionaries, to save cost (and improve accuracy?)  Maybe split Hebrew/Aramaic?

Testing:
* Check cases where the same wordform has different defintions.  If the definitions are equivalent, choose the best set, and merge the wordforms.


Timing:
* Do we want to wait for the new dictionaries to come online before doing this? 
