Architecture:
* Put resolver behind an endpoint
* Put resolver behind a queue?
* Put resolver behind a queue with a DB backed cache?
* On client side, iterate over segments and call resolver.  With returned results, adjust associated wordforms in Sefaria DB.

Potential:
* Allow search of just one or some dictionaries, to save cost (and improve accuracy?)  Maybe split Hebrew/Aramaic?
* Cache results wordform -> resolution
  * Create a separate process for evaluating words with high-probability definitions already identified.  
    or
  * Feed it into existing process as if it is already associated, and see if its accepted.

Testing:
* Check cases where the same wordform has different defintions.  Are both definitons ID'd and associated?

Timing:
* Do we want to wait for the new dictionaries to come online before doing this? 
